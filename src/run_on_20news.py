import argparse
import sys
import re
from collections import Counter
import numpy as np
from main import preprocess, estimate, inference


# load all docs
def load_text(input_txt_path):
    # ../data/datasets/profession/dataset.txt
    with open(input_txt_path, mode='r', encoding='utf-8') as text_file:
        # list , each element is an utterance
        text = list(map(lambda x: x.strip(), text_file.readlines()))
        text = [s.strip() for s in text]
    return text

def clean_html(str):
    clean_links = []
    left_mark, right_mark = '&lt;', '&gt;'
    # for every line find matching left_mark and nearest right_mark
    while True:
        next_left_start = str.find(left_mark)
        if next_left_start == -1:
            break
        next_right_start = str.find(right_mark, next_left_start)
        if next_right_start == -1:
            print("Right mark without Left: " + str)
            break
        # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
        clean_links.append(str[next_left_start: next_right_start + len(right_mark)])
        str = str[:next_left_start] + " " + str[next_right_start + len(right_mark):]
    # print(f"Cleaned {len(clean_links)} html links")
    return str

def clean_html1(str):
    pattern = re.compile(r'<[^>]+>',re.S)
    str = pattern.sub(' ',str)
    return str

# mainly for 20news
def clean_email(str):
    return " ".join([s for s in str.split(' ') if "@" not in s])

# clean one document/utterance
def clean_str(str):
    str = str.replace('.','')   # old, new
    str = clean_html(str)
    str = clean_html1(str)
    str = clean_email(str)

    str = re.sub(r"[^A-Za-z0-9(),\(\)=+.!?\"\']", " ", str)
    str = re.sub(r"\s{2,}", " ", str)
    return str.strip()

def load_clean_store(input_txt_path, cleaned_txt_path):
    cleaned_text = [clean_str(doc) for doc in load_text(input_txt_path)]
    with open(cleaned_txt_path, "w", encoding="utf-8") as f:
        for text in cleaned_text:
            f.write(text + '\n')
    print("load_clean_store over!")

def main(args):
    load_clean_store(args.input_txt_path, args.cleaned_txt_path)
    idx2word, doc_with_idx = preprocess(args.cleaned_txt_path)  # 236447
    theta, phi = estimate(args.num_of_topics, len(idx2word), args.alpha,    # 67006640 biterms
                          args.beta, args.niter, doc_with_idx, args.window_size)
    pz_d = inference(args.num_of_topics, doc_with_idx, theta, phi, args.window_size)
    print(pz_d.shape)
    for topic, times in list(Counter(np.argmax(pz_d, axis=0)).items()):
        print(topic, times)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Biterm Topic Model')
    parser.add_argument("--num-of-topics", type=int, default=5)     # 5 classes in 20news
    parser.add_argument("--alpha", type=float, default=2.5)  # round(50/num_of_topics, 3)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--niter", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--input-txt-path", type=str, default="../data/20news.txt")
    parser.add_argument("--cleaned-txt-path", type=str, default="../data/cleaned_20news.txt")
    parser.add_argument("--output-dir-path", type=str, default="../output/")
    args = parser.parse_args()

    # print(vars(args))
    main(args)