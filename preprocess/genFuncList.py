import os
import random
funcListPath = '../data/funcList.txt'
bcbPath = '../data/bigclonebenchdata/'

if __name__ == "__main__":
    funcList = []
    with open(funcListPath, 'w') as f:
        for ffile in len(os.listdir(bcbPath)):
            f.write(ffile+'\t')
            funcList.append(ffile)
