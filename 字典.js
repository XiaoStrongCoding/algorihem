/**
 * 字典树
 */

/**
 * 208. 实现 Trie (前缀树) https://leetcode.cn/problems/implement-trie-prefix-tree/description/?favorite=2cktkvj
 * 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。
 */

 var Trie = function() {
    this.children = {}
};

/** 
 * @param {string} word
 * @return {void}
 */
Trie.prototype.insert = function(word) {
    let node  = this.children
    for(const str of word) {
        if(!node[str]) {
            node[str] = {}
        }
        node = node[str]
    }
    node.isEnd = true
};

Trie.prototype.searchPrefix = function(word) {
    let node = this.children
    for(const str of word) {
        if(!node[str]) return false

        node = node[str]
    }
    return node
}

/** 
 * @param {string} word
 * @return {boolean}
 */
Trie.prototype.search = function(word) {
    const node = this.searchPrefix(word)

    return !!node && !!node.isEnd 
};

/** 
 * @param {string} prefix
 * @return {boolean}
 */
Trie.prototype.startsWith = function(prefix) {
    return this.searchPrefix(prefix)
};