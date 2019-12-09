from javalang.ast import Node
import javalang
import os

num = 0
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


def buildTree(rawTree):

    if not isinstance(rawTree, (str,Node)):
        return None, 0
    elif isinstance(rawTree, str):
        if rawTree == '' or rawTree[0] == '"' or rawTree[0] == "'":
            return None, 0
        astTree = {
            'name': str(rawTree),
            'children': []
        }
        return astTree, 1

    astTree = {
        'name': 'AstNode',
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

def getAstTree(filename):
    with open(filename, 'r') as f:
        data = f.read()
        rawTree = javalang.parse.parse_member_signature(data)
        root, nodeNum = buildTree(rawTree)
        root['name'] = nodeStr(rawTree)
    return root, nodeNum

if __name__ == "__main__":
    # f = open('../data/bigclonebenchdata/40044.txt', 'r')
    # data = f.read()
    # rawTree = javalang.parse.parse_member_signature(data)
    # # for child in rawTree.children:
    # #     print(child)
    root, nodeNum = getAstTree('../data/bigclonebenchdata/40044.txt')
    # root['name'] = 'MethodDeclaration'
    print(root)
    print(nodeNum)
    # f.close()
