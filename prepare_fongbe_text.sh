#!/usr/bin/env bash
# Custom Fongbe text preparation script
# Creates the same output format as the official prepare_text.sh but uses G2P dictionary for Fongbe

text_corpus=$1
target_dir=$2
g2p_dict=$3
min_phones=${4:-10}
sil_prob=${5:-0.5}

if [ -z "$text_corpus" ] || [ -z "$target_dir" ] || [ -z "$g2p_dict" ]; then
    echo "Usage: $0 <text_corpus> <target_dir> <g2p_dict> [min_phones] [sil_prob]"
    echo "Example: $0 fongbe_data/combined_text.txt fongbe_text_features codes/dictionary.txt 10 0.5"
    exit 1
fi

echo "🔤 Preparing Fongbe text data with G2P dictionary..."
echo "   Text corpus: $text_corpus"
echo "   Target dir: $target_dir"
echo "   G2P dict: $g2p_dict"
echo "   Min phones: $min_phones"
echo "   Silence prob: $sil_prob"

mkdir -p $target_dir

# Step 1: Normalize and filter text (similar to official script)
echo "📝 Step 1: Normalizing text..."
python3 -c "
import sys
import re
import os

def normalize_fongbe_text(text):
    # Basic text normalization for Fongbe
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation except apostrophes (important for Fongbe)
    text = re.sub(r'[^\w\s\']', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

with open('$text_corpus', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            normalized = normalize_fongbe_text(line)
            if normalized and len(normalized.split()) >= 2:  # Keep lines with at least 2 words
                print(normalized)
" > $target_dir/lm.upper.lid.txt

echo "✅ Normalized $(wc -l < $target_dir/lm.upper.lid.txt) lines of text"

# Step 2: Create word dictionary (similar to official fairseq preprocessing)
echo "📚 Step 2: Creating word dictionary..."
python3 -c "
import sys
from collections import Counter

# Count word frequencies
word_counts = Counter()
with open('$target_dir/lm.upper.lid.txt', 'r') as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            word_counts[word] += 1

# Create fairseq-style dictionary (word freq format)
with open('$target_dir/dict.txt', 'w') as f:
    for word, count in word_counts.most_common():
        if count >= 2:  # threshold similar to official script
            print(f'{word} {count}', file=f)

# Extract just the words
with open('$target_dir/words.txt', 'w') as f:
    for word, count in word_counts.most_common():
        if count >= 2:
            print(word, file=f)
"

echo "✅ Created word dictionary with $(wc -l < $target_dir/words.txt) unique words"

# Step 3: Convert words to phonemes using G2P dictionary
echo "🔊 Step 3: Converting words to phonemes using G2P dictionary..."
python3 -c "
import sys
import re

def load_g2p_dictionary(dict_path):
    g2p_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                grapheme, phoneme = line.strip().split('\t', 1)
                g2p_dict[grapheme] = phoneme
    return g2p_dict

def apply_g2p_rules(word, g2p_dict):
    # Handle nasalization rules
    word = re.sub(r'([aeiouɛɔ])n([ptk])', r'\1̃\2', word)
    word = re.sub(r'([aeiouɛɔ])m([pb])', r'\1̃\2', word)
    word = re.sub(r'([aeiouɛɔ])n$', r'\1̃', word)
    word = re.sub(r'([aeiouɛɔ])m$', r'\1̃', word)
    
    result = []
    i = 0
    while i < len(word):
        found = False
        for length in range(min(3, len(word) - i), 0, -1):
            substr = word[i:i+length]
            if substr in g2p_dict:
                result.append(g2p_dict[substr])
                i += length
                found = True
                break
        if not found:
            result.append(word[i])
            i += 1
    return ' '.join(result)

# Load G2P dictionary
g2p_dict = load_g2p_dictionary('$g2p_dict')

# Convert words to phones
with open('$target_dir/words.txt', 'r') as f_in, open('$target_dir/phones.txt', 'w') as f_out:
    for line in f_in:
        word = line.strip()
        phones = apply_g2p_rules(word, g2p_dict)
        print(phones, file=f_out)
"

echo "✅ Converted words to phonemes"

# Step 4: Create lexicon (word -> phone mapping)
echo "📖 Step 4: Creating lexicon..."
paste $target_dir/words.txt $target_dir/phones.txt > $target_dir/lexicon.lst
echo "✅ Created lexicon with $(wc -l < $target_dir/lexicon.lst) entries"

# Step 5: Create phone dictionary and filter lexicon
echo "📱 Step 5: Creating phone dictionary..."
mkdir -p $target_dir/phones

python3 -c "
from collections import Counter
import sys

# Count phone frequencies
phone_counts = Counter()
with open('$target_dir/phones.txt', 'r') as f:
    for line in f:
        phones = line.strip().split()
        for phone in phones:
            phone_counts[phone] += 1

# Create phone dictionary (keeping phones with min frequency)
with open('$target_dir/phones/dict.txt', 'w') as f:
    for phone, count in phone_counts.most_common():
        if count >= $min_phones:
            print(f'{phone} {count}', file=f)

# Filter lexicon to only include phones in dictionary
valid_phones = set()
with open('$target_dir/phones/dict.txt', 'r') as f:
    for line in f:
        phone = line.strip().split()[0]
        valid_phones.add(phone)

with open('$target_dir/lexicon.lst', 'r') as f_in, open('$target_dir/lexicon_filtered.lst', 'w') as f_out:
    for line in f_in:
        items = line.strip().split()
        if len(items) >= 2:
            phones = items[1:]
            # Check if all phones are valid
            if all(phone in valid_phones for phone in phones):
                print(line.strip(), file=f_out)
"

echo "✅ Created phone dictionary with $(wc -l < $target_dir/phones/dict.txt) unique phones"

# Step 6: Convert text corpus to phonemes with silence insertion
echo "🔇 Step 6: Converting text corpus to phonemes with silence..."
python3 -c "
import random
import sys
import re

def load_g2p_dictionary(dict_path):
    g2p_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                grapheme, phoneme = line.strip().split('\t', 1)
                g2p_dict[grapheme] = phoneme
    return g2p_dict

def apply_g2p_rules(word, g2p_dict):
    # Handle nasalization rules
    word = re.sub(r'([aeiouɛɔ])n([ptk])', r'\1̃\2', word)
    word = re.sub(r'([aeiouɛɔ])m([pb])', r'\1̃\2', word)
    word = re.sub(r'([aeiouɛɔ])n$', r'\1̃', word)
    word = re.sub(r'([aeiouɛɔ])m$', r'\1̃', word)
    
    result = []
    i = 0
    while i < len(word):
        found = False
        for length in range(min(3, len(word) - i), 0, -1):
            substr = word[i:i+length]
            if substr in g2p_dict:
                result.append(g2p_dict[substr])
                i += length
                found = True
                break
        if not found:
            result.append(word[i])
            i += 1
    return ' '.join(result)

# Load valid phones
valid_phones = set()
with open('$target_dir/phones/dict.txt', 'r') as f:
    for line in f:
        phone = line.strip().split()[0]
        valid_phones.add(phone)

# Load lexicon for quick lookup
word_to_phones = {}
with open('$target_dir/lexicon_filtered.lst', 'r') as f:
    for line in f:
        items = line.strip().split()
        if len(items) >= 2:
            word = items[0]
            phones = ' '.join(items[1:])
            word_to_phones[word] = phones

g2p_dict = load_g2p_dictionary('$g2p_dict')
sil_prob = $sil_prob

with open('$target_dir/lm.upper.lid.txt', 'r') as f_in, open('$target_dir/phones/lm.phones.filtered.txt', 'w') as f_out:
    for line in f_in:
        words = line.strip().split()
        phone_seq = []
        
        # Add initial silence with probability
        if random.random() < sil_prob:
            phone_seq.append('<SIL>')
            
        for i, word in enumerate(words):
            # Convert word to phones
            if word in word_to_phones:
                phones = word_to_phones[word].split()
            else:
                phones = apply_g2p_rules(word, g2p_dict).split()
            
            # Filter phones to only include valid ones
            phones = [p for p in phones if p in valid_phones]
            phone_seq.extend(phones)
            
            # Add silence between words with probability
            if i < len(words) - 1 and random.random() < sil_prob:
                phone_seq.append('<SIL>')
        
        # Add final silence with probability  
        if random.random() < sil_prob:
            phone_seq.append('<SIL>')
            
        if phone_seq:  # Only output if we have phones
            print(' '.join(phone_seq), file=f_out)
"

echo "✅ Converted text corpus to phonemes with silence insertion"

# Step 7: Create final phone dictionary with silence and preprocess
echo "📄 Step 7: Creating final phone dictionary..."
cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt

# Use fairseq preprocessing to create binary files
PYTHONPATH=$FAIRSEQ_ROOT python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py \
    --dataset-impl mmap \
    --trainpref $target_dir/phones/lm.phones.filtered.txt \
    --workers 10 \
    --only-source \
    --destdir $target_dir/phones \
    --srcdict $target_dir/phones/dict.phn.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to preprocess phone data with fairseq"
    exit 1
fi

echo ""
echo "✅ Fongbe text preparation complete!"
echo ""
echo "📊 Output files:"
echo "   • Text corpus: $target_dir/lm.upper.lid.txt"
echo "   • Word dict: $target_dir/dict.txt"
echo "   • Phone dict: $target_dir/phones/dict.phn.txt"  
echo "   • Lexicon: $target_dir/lexicon_filtered.lst"
echo "   • Phone corpus: $target_dir/phones/lm.phones.filtered.txt"
echo "   • Fairseq binary: $target_dir/phones/train.bin"
echo ""
echo "📈 Statistics:"
echo "   • Words: $(wc -l < $target_dir/words.txt)"
echo "   • Phones: $(wc -l < $target_dir/phones/dict.txt)"
echo "   • Text lines: $(wc -l < $target_dir/phones/lm.phones.filtered.txt)"