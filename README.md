# SAT_project
> Software clone , aiming at identifying out code fragments with similar functionalities, has played an important role in software maintenance and evolution.
## TBCCD: A method of code clone  
TBCDD is a method proposed for detecting code clones in this [paper](http://taoxie.cs.illinois.edu/publications/icpc19-clone.pdf), where a Tree-based CNN is introduced into this task. In this method, the ASTs of Java/C source codes are generated. Then words of all nodes in ASTs are embedded by a combined character-wise one-hot encoder. Based on the embedding, Tree-based CNNs are employed to extract Ast-level representation. At last, the model predict whether a pair of codes are cloned ones by comparing their features' cosine similarity. Here we newly implement a pytorch version of TBCDD, which has been proved powerful in detecting cloned codes.

## How to use our pytorch version TBCDD
Firstly, you need to get similarity matrix and dataset in [Baidu Netdisk]() with code    . You should put them into /data/ and unzip the dataset.

Then, you have to generate your own train/val/test datasets by directly running 
```
python ./preprocess/genFuncList.py
```

Having prepared datasets, you can run 

```
python ./src/train_model.py
```
to train the TBCDD model and see results in test sets.

## required packages
python3.7, pytorch1.1.0, javalang, pickle, numpy
