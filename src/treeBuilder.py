from javalang.ast import Node
import javalang
import os
from queue import Queue


def nodeStr(rawNode):
    return type(rawNode).__name__


def unfoldList(childList):
    returnList = []
    if not isinstance(childList, (tuple, list)):
        return [childList]
    for child in childList:
        if isinstance(child, (list, tuple)):
            returnList.extend(unfoldList(child))
        else:
            returnList.append(child)
    return returnList


def buildTree(rawTree, withId=True):

    if not isinstance(rawTree, (str,Node)):
        return None, 0
    elif isinstance(rawTree, str):
        if rawTree == '' or rawTree[0] == '"' or rawTree[0] == "'" or rawTree.startswith("/*"):
            return None, 0
        astTree = {
            'name': str(rawTree),
            'children': []
        }
        return astTree, 1

    if withId:
        name = nodeStr(rawTree)
    else:
        name = 'AstNode'

    astTree = {
        'name': name,
        'children': []
    }
    nodeNum = 1

    children = unfoldList(rawTree.children)
    for child in children:
        subTree, subNodeNum = buildTree(child)
        if subTree is None:
            continue
        astTree['children'].append(subTree)
        nodeNum += subNodeNum

    return astTree, nodeNum


def getAstNodeList(root, withId=True):
    # print(root, type(root))
    if '\n' in root['name']:
        print(root['name'])
    nodeList = [root['name']]
    for child in root['children']:
        nodeList.extend(getAstNodeList(child, withId=True))
    return nodeList


def getChildrenList(root):
    wordList = []
    childrenList = []
    nodeQ = []
    nodeQ.append((root,-1))
    while len(nodeQ) > 0:
        root, parentID = nodeQ.pop(0)
        childrenList.append([])
        wordList.append(root['name'])
        ind = len(childrenList) - 1
        for child in root['children']:
            nodeQ.append((child, ind))
        if parentID >= 0:
            childrenList[parentID].append(ind)
    return childrenList, wordList


def getAstTree(filename):
    with open(filename, 'r') as f:
        data = f.read()
        rawTree = javalang.parse.parse_member_signature(data)
        root, nodeNum = buildTree(rawTree)
    return root, nodeNum

if __name__ == "__main__":
    # f = open('../data/bigclonebenchdata/40044.txt', 'r')
    # data = f.read()
    # rawTree = javalang.parse.parse_member_signature(data)
    # # for child in rawTree.children:
    # #     print(child)
    root, nodeNum = getAstTree('../data/bigclonebenchdata/5494012.txt')
    # print(root)
    clist, wlist = getChildrenList(root)
    print(clist)
    # print(wlist)
    # root['name'] = 'MethodDeclaration'
    # nodeList = getAstNodeList(root)
    # print(nodeList)
    # print(nodeNum)
    # f.close()
