
# Task 4: Lattice-Based WER Evaluation

from collections import Counter

def tokenize(text):
    return text.strip().split()

def align_sequences(sequences):
    max_len = max(len(seq) for seq in sequences)
    aligned = []
    for seq in sequences:
        new_seq = seq.copy()
        while len(new_seq) < max_len:
            new_seq.append("<blank>")
        aligned.append(new_seq)
    return aligned

def build_lattice(reference, model_outputs):
    all_sequences = [reference] + model_outputs
    tokenized = [tokenize(seq) for seq in all_sequences]
    aligned = align_sequences(tokenized)

    lattice = []
    for i in range(len(aligned[0])):
        words = set()
        for seq in aligned:
            if seq[i] != "<blank>":
                words.add(seq[i])
        lattice.append(list(words))
    return lattice

def normalize_word(word):
    mapping = {
        "चौदह": "14",
        "14": "14",
        "किताबें": "किताब",
        "किताबे": "किताब"
    }
    return mapping.get(word, word)

def apply_majority_voting(lattice, model_outputs):
    tokenized_outputs = [tokenize(seq) for seq in model_outputs]
    aligned = align_sequences(tokenized_outputs)

    updated = []
    for i in range(len(lattice)):
        words = []
        for seq in aligned:
            if seq[i] != "<blank>":
                words.append(seq[i])
        count = Counter(words)
        for word, freq in count.items():
            if freq >= 2:
                lattice[i].append(word)
        updated.append(list(set(lattice[i])))
    return updated

def lattice_wer(prediction, lattice):
    pred_tokens = tokenize(prediction)
    errors = 0
    total = len(lattice)

    for i in range(total):
        if i >= len(pred_tokens):
            errors += 1
            continue

        pred_word = normalize_word(pred_tokens[i])
        lattice_words = [normalize_word(w) for w in lattice[i]]

        if pred_word not in lattice_words:
            errors += 1

    return errors / total

def normal_wer(prediction, reference):
    pred = tokenize(prediction)
    ref = tokenize(reference)

    errors = 0
    total = len(ref)

    for i in range(total):
        if i >= len(pred) or pred[i] != ref[i]:
            errors += 1

    return errors / total


if __name__ == "__main__":
    reference = "उसने चौदह किताबें खरीदीं"

    model_outputs = [
        "उसने 14 किताबें खरीदी",
        "उसने चौदह किताबे खरीदीं",
        "उसने चौदह किताबें खरीदी",
        "उसने 14 किताबे खरीदी",
        "उसने चौदह किताबें खरीदीं"
    ]

    lattice = build_lattice(reference, model_outputs)
    lattice = apply_majority_voting(lattice, model_outputs)

    print("\nLattice:")
    for i, bin_words in enumerate(lattice):
        print(f"Position {i}: {bin_words}")

    print("\nWER Comparison:")
    for i, pred in enumerate(model_outputs):
        print(f"\nModel {i+1}:")
        print("Normal WER:", round(normal_wer(pred, reference), 3))
        print("Lattice WER:", round(lattice_wer(pred, lattice), 3))
