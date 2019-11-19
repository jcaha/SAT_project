# SAT_project
Our project aims to provide a deep learning solution for code clone detection.
## What is Software clone
软件系统中克隆代码的检测与管理是软件工程中的基本问题之一。

程序员为了提高开发效率经常对软件系统的源代码进行拷贝粘贴及修改，这种重用机制通常会导致在源代码中出现很多相同或相似的代码段，这类代码段被称为**克隆代码**。

## Why should we pay attention to cloned codes
- 克隆代码导致源代码的规模增大即冗余，增加了资源的需求，这对嵌入式系统和手持设备的影响尤为明显。
- 克隆一段含有未知BUG的代码,可能会导致BUG的繁育。
- 维护者修改一段代码时,需对这段代码的所有克隆进行一致的修改,若修改不一致则会引入新的BUG。
为了更好地解决这些弊端，我们需要对系统进行克隆代码检测。

## What is code clone detection
> Software clone detection, aiming at identifying out code fragments with similar functionalities, has played an important role in software maintenance and evolution.

## dataset for deep learning
BigCloneBench in [Baidu Netdisk](https://pan.baidu.com/s/1DK6XJmfj_oKWqDvkm0ehlQ) with code vffp.
