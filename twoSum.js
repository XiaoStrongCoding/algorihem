/**
 * 双指针遍历 例：三数之和等于0
 * 滑动窗口   例：最长无重复子串
 * 盛水题
 * 括号题
 * 排列/组合题
 */

/**
 * 1. 两数之和 https://leetcode.cn/problems/two-sum/?favorite=2cktkvj
 * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，
 * 并返回它们的数组下标。
   你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
   你可以按任意顺序返回答案。
 */

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
 var twoSum = function(nums, target) {
    const map = new Map()

    for(let i=0; i<nums.length; i++) {
        if(map.has(target-nums[i])){
            return [map.get(target-nums[i]), i]
        } else {
            map.set(nums[i], i)
        }
    }
};

/**
 * 15. 三数之和 https://leetcode.cn/problems/3sum/description/
 * 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，
 * 同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
    你返回所有和为 0 且不重复的三元组。
    注意：答案中不可以包含重复的三元组。
 */

/**
 * @param {number[]} nums
 * @return {number[][]}
 */
 var threeSum = function(nums) {
    nums = nums.sort((a,b) => a-b)
    const result = []
    const n = nums.length
    for(let i = 0; i<n; i++) {
        if(nums[i]>0) break;
        if(i&&nums[i]==nums[i-1]) continue;

        let l = i+1;
        let r = n-1;
        while(l<r) {
            const sum = nums[i] + nums[l] + nums[r]

            if(sum === 0) {
                result.push([nums[i], nums[l], nums[r]])
                while(l<r&&nums[l]===nums[l+1]) l++
                while(l<r&&nums[r]===nums[r-1]) r--
                l++
                r--
            } else if(sum>0) {
                r--
            } else {
                l++
            }
        }
    }

    return result
};

/**
 * 3. 无重复字符的最长子串 https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/
 * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
 */

/**
 * @param {string} s
 * @return {number}
 */
 var lengthOfLongestSubstring = function(s) {
    const set = new Set()
    const n = s.length
    let rk = -1
    let ans = 0
    for(let i=0; i<n; i++) {
        if(i) {
            set.delete(s[i-1])
        }
        while(rk+1<n&&!set.has(s[rk+1])) {
            rk++
            set.add(s[rk])
        }
        ans = Math.max(ans, rk-i+1)
    }
    return ans
};

/**
 * 76. 最小覆盖子串 https://leetcode.cn/problems/minimum-window-substring/description/?favorite=2cktkvj
 * 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
 */

/**
 * @param {string} s
 * @param {string} t
 * @return {string}
 */
 var minWindow = function(s, t) {
    let minLen = s.length + 1
    let start = s.length
    let map = {}, missingType = 0
    for(const str of t) {
        if(!map[str]) {
            missingType++;
            map[str]=1;
        } else {
            map[str]++
        }
    }

    let l =0, r = 0
    for(; r<s.length; r++) {
        const rightChar = s[r];
        if(map[rightChar] !== undefined) map[rightChar]--
        if(map[rightChar]==0) missingType--;

        while(missingType==0) {
            if(r-l+1<minLen) {
                minLen = r-l+1
                start = l
            }

            let leftChar = s[l];
            if(map[leftChar] !== undefined) map[leftChar]++;
            if(map[leftChar]>0) missingType++;
            l++;
        }
    }

    if(start==s.length) return '';
    return s.substring(start, start + minLen)
};



/**
 * 5. 最长回文子串 https://leetcode.cn/problems/longest-palindromic-substring/description/
 * 给你一个字符串 s，找到 s 中最长的回文子串。
   如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
 */

/**
 * @param {string} s
 * @return {string}
 */
 var longestPalindrome = function(s) {
    if(s.length<2) return s

    let res = ''

    for(let i=0; i<s.length; i++) {
        helper(i, i)
        helper(i, i+1)
    }

    function helper(start, end) {
        while(start>=0 && end<s.length && s[start]===s[end]) {
            start--
            end++
        }

        if(end-start-1 > res.length) {
            res = s.slice(start+1, end)
        }
    }

    return res
};

/**
 * 17. 电话号码的字母组合 https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/?favorite=2cktkvj
 * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合
 */

/**
 * @param {string} digits
 * @return {string[]}
 */
 var letterCombinations = function(digits) {
    if(digits.length==0) return []
    const map = { '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz' };
    const result = []

    function dfs(str, layer) {
        if(layer > digits.length-1) {
            result.push(str)
            return
        }
        const letter = map[digits[layer]]
        for(let s of letter) {
            dfs(str+s, layer+1)
        }
    }

    dfs('', 0);
    return result
};

/**
 * 22. 括号生成 https://leetcode.cn/problems/generate-parentheses/description/
 * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
 */

/**
 * @param {number} n
 * @return {string[]}
 */
 var generateParenthesis = function(n) {
    const res = []
    function dfs(lRemain, rRemian, str) {
        if(str.length === 2*n) {
            res.push(str)
            return
        }

        if(lRemain) {
            dfs(lRemain-1, rRemian, str+'(')
        }

        if(lRemain<rRemian) {
            dfs(lRemain, rRemian-1, str+')')
        }
    }

    dfs(n,n,'')
    return res
};

/**
 * 20. 有效的括号 https://leetcode.cn/problems/valid-parentheses/description/
 * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
 */

/**
 * @param {string} s
 * @return {boolean}
 */
 var isValid = function(s) {
    if(s.length%2===1) return false
    const map = {']': '[', ')':'(', '}':'{'}
    const result = []

    for(const str of s) {
        if(map[str]) {
            if(result.pop() !== map[str]) return false
        } else {
            result.push(str)
        }
    }

    return !result.length
};

/**
 * 32. 最长有效括号 https://leetcode.cn/problems/longest-valid-parentheses/description/
 * 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
 */

/**
 * @param {string} s
 * @return {number}
 */
 var longestValidParentheses = function(s) {
    if(s.length<2) return 0
    let ans = 0
    let stact = [-1]

    for(let i=0; i<s.length; i++) {
        if(s[i] === '(') {
            stact.push(i)
        } else {
            stact.pop()

            const n = stact.length
            if(!n) {
                stact.push(i)
            } else {
                ans = Math.max(ans, i-stact[n-1])
            }
        }
    }

    return ans
};

/**
 * 31. 下一个排列 https://leetcode.cn/problems/next-permutation/description/
 */

/**
 * @param {number[]} nums
 * @return {void} Do not return anything, modify nums in-place instead.
 */
 var nextPermutation = function(nums) {
    let l = nums.length - 2
    while(l>=0 && nums[l] >= nums[l+1]) {
        l--
    }

    if(l>=0) {
        let j = nums.length - 1
        while(j>=0&&nums[j]<=nums[l]) {
            j--
        }
        [nums[l], nums[j]] = [nums[j], nums[l]]
    }

    l+=1
    let r = nums.length - 1
    while(l<r) {
        [nums[l], nums[r]] = [nums[r], nums[l]]
        l++
        r--
    }

    return nums
};


/****************************排列/组合************************************/

/**
 * 39. 组合总和 https://leetcode.cn/problems/combination-sum/description/
 * 给你一个 无重复元素 的整数数组
 */

/**
 * @param {number[]} candidates
 * @param {number} target
 * @return {number[][]}
 */
 var combinationSum = function(candidates, target) {
    const result = [], path = []
    candidates.sort()
    function dfs(i, sum) {
        if(sum > target) return

        if(sum === target) {
            result.push([...path])
            return
        }

        for(let j=i; j<candidates.length; j++) {
            if(candidates[j] > target - sum) continue

            path.push(candidates[j])
            dfs(j, sum+candidates[j])
            path.pop()
        }
    }

    dfs(0,0)
    return result
};

/**
 * 46. 全排列 https://leetcode.cn/problems/permutations/description/
 * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
 */

/**
 * @param {number[]} nums
 * @return {number[][]}
 */
 var permute = function(nums) {
    const result = [], path = []
    const n = nums.length
    function dfs(used) {
        if(path.length === n) {
            result.push([...path])
        }

        for(let i=0; i<n; i++) {
            if(used[i]) continue;
            path.push(nums[i])
            used[i] = true
            dfs(used)
            path.pop()
            used[i] = false
        }
    }
    dfs([])
    return result
};

/**
 * 78. 子集 https://leetcode.cn/problems/subsets/?favorite=2cktkvj
 * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
    解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
 */

/**
 * @param {number[]} nums
 * @return {number[][]}
 */
var subsets = function(nums) {
    const res = [], cur = []
    function dfs(i, ) {
        res.push(cur.slice())
        for(let j=i; j<nums.length; j++) {
            cur.push(nums[j])
            dfs(j+1)
            cur.pop()
        }
    }
    dfs(0,[])
    return res
};

/**
 * 131. 分割回文串
 * 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串。返回 s 所有可能的分割方案。
 * 输入：s = "aab"
   输出：[["a","a","b"],["aa","b"]]
 */

/**
 * @param {string} s
 * @return {string[][]}
 */
 var partition = function(s) {
    let n = s.length
    const ans = [], path=[]
    function dfs(i, start) {
        if(i===n) {
            ans.push(path.slice())
            return
        }
        if(i<n-1) {
            dfs(i+1, start)
        }

        if(isPalindrom(s, start, i)) {
            path.push(s.substring(start, i+1))
            dfs(i+1, i+1)
            path.pop()
        }

    }
    dfs(0,0)
    return ans
};

function isPalindrom(s, l, r) {
    while(l<r) {
        if(s[l++]!==s[r--]) return false
    }
    return true
}

/**
 * N皇后 https://leetcode.cn/problems/eight-queens-lcci/description/?orderBy=most_votes&languageTags=javascript
 * 设计一种算法，打印 N 皇后在 N × N 棋盘上的各种摆法，其中每个皇后都不同行、不同列，也不在对角线上。这里的“对角线”指的是所有的对角线，不只是平分整个棋盘的那两条对角线。
 */

/**
 * @param {number} n
 * @return {string[][]}
 */
 var solveNQueens = function(n) {
    const dp = Array.from(new Array(n), () => new Array(n).fill('.'));
    const ans = []
    function isVaild(row, col) {
        for(let i=0; i<row; i++) {
            for(let j=0; j<n; j++) {
                if(dp[i][j] === 'Q' && (j==col || i+j === row+col || i-j === row-col)) {
                    return false
                }
            }
        }
        return true
    }

    function dfs(row) {
        if(row === n) {
            const stringArr = dp.slice()
            for(let i=0; i<n; i++) {
                stringArr[i] = stringArr[i].join('')
            }
            ans.push(stringArr)
            return
        } else {
            for(let col=0; col<n; col++) {
                if(isVaild(row,col)) {
                    dp[row][col] = 'Q'
                    dfs(row+1)
                    dp[row][col] = '.'
                }
            }
        }
    }
    dfs(0)
    return ans
};

/*************************************************排列/组合 end*************************/

/**
 * 56. 合并区间 https://leetcode.cn/problems/merge-intervals/description/ 代码精简
 * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，
 * 并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
 */

/**
 * @param {number[][]} intervals
 * @return {number[][]}
 */
 var merge = function(intervals) {
    if(intervals.length===1) return intervals
    intervals.sort((a,b)=>a[0]-b[0])
    const result = []
    let pre = intervals[0]
    for(let i=1; i<intervals.length; i++) {
        const cur = intervals[i]
        if(cur[0] > pre[1]) {
            result.push(pre)
            pre = cur
        } else {
            pre[1] = Math.max(pre[1], cur[1])
        }
    }
    result.push(pre)
    return result
};

/**
 * 75. 颜色分类 https://leetcode.cn/problems/sort-colors/description/
 * 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
    我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
    必须在不使用库内置的 sort 函数的情况下解决这个问题。
 */

/**
 * @param {number[]} nums
 * @return {void} Do not return anything, modify nums in-place instead.
 */
 var sortColors = function(nums) {
    let l = 0, r=nums.length-1, i=0

    while(i<=r) {
        if(nums[i] === 0) {
            swap(nums, i, l);
            l++;
            i++;
        } else if(nums[i]===2) {
            swap(nums, i, r);
            r--
        } else {
            i++
        }
    }

    return nums
};

function swap(arr,a,b){
    let tmp = arr[a];
    arr[a] = arr[b]
    arr[b] = tmp;
}

/**
 * 48. 旋转图像 https://leetcode.cn/problems/rotate-image/description/
 * 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
 */

/**
 * @param {number[][]} matrix
 * @return {void} Do not return anything, modify matrix in-place instead.
 */
 var rotate = function(matrix) {
    const n = matrix.length
    for(let i = 0; i<Math.floor(n/2); i++) {
        for(let j = 0; j<Math.floor((n+1)/2); j++) {
            const temp = matrix[i][j];
            matrix[i][j] = matrix[n - j - 1][i];
            matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
            matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
            matrix[j][n - i - 1] = temp;
        }
    }
};

/**
 * 79. 单词搜索 https://leetcode.cn/problems/word-search/description/?favorite=2cktkvj
 * 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
    单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
 */

/**
 * @param {character[][]} board
 * @param {string} word
 * @return {boolean}
 */
 var exist = function(board, word) {
    board[-1] = []
    board.push([])

    for(let x=0; x<board.length-1; x++) {
        for(let y=0; y<board[0].length; y++) {
            if(board[x][y]===word[0] && dfs(board,word,x,y,0)) return true
        }
    }

    return false
};

function dfs(board,word,x,y,index) {
    if(index+1 === word.length) return true

    const temp = board[x][y]
    board[x][y] = false

    if(board[x+1][y]===word[index+1]&&dfs(board,word,x+1,y,index+1)) return true
    if(board[x][y+1]===word[index+1]&&dfs(board,word,x,y+1,index+1)) return true
    if(board[x-1][y]===word[index+1]&&dfs(board,word,x-1,y,index+1)) return true
    if(board[x][y-1]===word[index+1]&&dfs(board,word,x,y-1,index+1)) return true

    board[x][y] = temp
}

/*******************哈希表 start******************************/


/**
 * 49. 字母异位词分组 https://leetcode.cn/problems/group-anagrams/description/
 * 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
 * 字母异位词 是由重新排列源单词的所有字母得到的一个新单词。
 * 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
 * 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
 */
/**
 * @param {string[]} strs
 * @return {string[][]}
 */
 var groupAnagrams = function(strs) {
    if(strs.length===1) return [strs]

    const map = new Map()

    strs.forEach(str => {
        const key = str.split('').sort().join('')
        let list = map.get(key) ? map.get(key) : new Array();
        list.push(str);
        map.set(key, list);
    })

    return Array.from(map.values())
};

/**
 * 128. 最长连续序列 https://leetcode.cn/problems/longest-consecutive-sequence/description/?favorite=2cktkvj
 * 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
   请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var longestConsecutive = function(nums) {
    const set = new Set(nums)
    let maxCount = 0
    for(let num of nums) {
        if(!set.has(num-1)) { // 优化算法，从不连续头开始
            let cur = num
            let count = 1
            while(set.has(cur+1)) {
                cur++
                count++
            }
            maxCount = Math.max(maxCount, count)
        }
    }
    return maxCount
};

 var longestConsecutive = function(nums) {
    const map = new Map()
    let maxCount = 0
    for(const num of nums) {
        if(!map.has(num)) { // 去重
            const prelen = map.get(num-1) || 0
            const nextlen = map.get(num+1) || 0
            const totalLen = 1 + prelen + nextlen
            map.set(num, totalLen)
            maxCount = Math.max(maxCount, totalLen)
            map.set(num-prelen, totalLen)
            map.set(num+nextlen, totalLen)
        }
    }
    return maxCount
};
/*******************哈希表 end******************************/

/*******************位运算 start******************************/

/**
 * 136. 只出现一次的数字 https://leetcode.cn/problems/single-number/description/?favorite=2cktkvj
 * 给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var singleNumber = function(nums) {
    let ans = 0
    for(const num of nums) {
        ans ^= num
    }
    return ans
};

/*******************位运算 end******************************/

/*********************堆栈 start*******************************/

/**
 * 155. 最小栈 https://leetcode.cn/problems/min-stack/description/?favorite=2cktkvj
 * 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
    实现 MinStack 类:
        MinStack() 初始化堆栈对象。
        void push(int val) 将元素val推入堆栈。
        void pop() 删除堆栈顶部的元素。
        int top() 获取堆栈顶部的元素。
        int getMin() 获取堆栈中的最小元素。
 */

var MinStack = function() {
    this.min_stack = [Infinity]
    this.stack = []
};

/** 
 * @param {number} val
 * @return {void}
 */
MinStack.prototype.push = function(val) {
    this.stack.push(val)
    this.min_stack.push(Math.min(this.min_stack[this.min_stack.length-1], val))
};

/**
 * @return {void}
 */
MinStack.prototype.pop = function() {
    this.stack.pop()
    this.min_stack.pop()
};

/**
 * @return {number}
 */
MinStack.prototype.top = function() {
    return this.stack[this.stack.length-1]
};

/**
 * @return {number}
 */
MinStack.prototype.getMin = function() {
    return this.min_stack[this.min_stack.length-1]
};

/*********************堆栈 end*******************************/

/**
 * 169. 多数元素 https://leetcode.cn/problems/majority-element/description/?favorite=2cktkvj
 * 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
   你可以假设数组是非空的，并且给定的数组总是存在多数元素。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var majorityElement = function(nums) {
    let ans = 0
    let count = 0
    for(const num of nums) {
        if(count === 0) ans = num
        count += ans === num ? 1 : -1
    }

    return ans
};

/**
 * 238. 除自身以外数组的乘积 https://leetcode.cn/problems/product-of-array-except-self/description/?favorite=2cktkvj
 * 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
 * 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
 */

/**
 * @param {number[]} nums
 * @return {number[]}
 */
 var productExceptSelf = function(nums) {
    const n = nums.length
    const output = []
    output[0] = 1
    for(let i=1; i<n; i++) {
        output[i] = output[i-1]*nums[i-1]
    }

    let right = 1
    for(let i=n-1; i>=0; i--) {
        output[i] *= right;
        right *= nums[i]
    }
    return output
};

/**
 * 从不同角度（右上角）
 * 240. 搜索二维矩阵 II https://leetcode.cn/problems/search-a-2d-matrix-ii/description/
 * 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
    每行的元素从左到右升序排列。
    每列的元素从上到下升序排列。
 */

/**
 * @param {number[][]} matrix
 * @param {number} target
 * @return {boolean}
 */
 var searchMatrix = function(matrix, target) {
    const row = matrix.length, col = matrix[0].length
    let x = 0, y = col-1

    while(x<row&&y>=0) {
        if(matrix[x][y] === target) {
            return true
        } else if(matrix[x][y] > target) {
            y--
        } else {
            x++
        }
    }

    return false
};

/*******************双指针 start******************************/

/**
 * 283. 移动零 https://leetcode.cn/problems/move-zeroes/description/?favorite=2cktkvj
 * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
   请注意 ，必须在不复制数组的情况下原地对数组进行操作。
 */

/**
 * @param {number[]} nums
 * @return {void} Do not return anything, modify nums in-place instead.
 */
 var moveZeroes = function(nums) {
    return nums.sort((a,b)=>b ? 0 : -1)
    /**
     * 如果 compareFn(a, b) 大于 0，b 会被排列到 a 之前。
       如果 compareFn(a, b) 小于 0，那么 a 会被排列到 b 之前；
       如果 compareFn(a, b) 等于 0，a 和 b 的相对位置不变。
     */
};

var moveZeroes = function(nums) {
    const n = nums.length
    let l = 0, r = 0
    while(r<n) {
        if(nums[r]) {
            [nums[l], nums[r]] = [nums[r], nums[l]]
            l++
        }
        r++
    }
    return nums
};

// 暴力 nums.splice(i, 1)   nums.push(0)


/**
 * 11. 盛最多水的容器 https://leetcode.cn/problems/container-with-most-water/description/
 */

/**
 * @param {number[]} height
 * @return {number}
 */
 var maxArea = function(height) {
    let ans = 0
    for(let l=0, r=height.length-1; l<r;) {
        const minHeight = height[l]>height[r] ? height[r--] : height[l++];
        const area =  minHeight * (r-l+1)
        ans = Math.max(ans, area)
    }
    return ans
};

/**
 * 42. 接雨水 https://leetcode.cn/problems/trapping-rain-water/description/
 * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
 */

/**
 * @param {number[]} height
 * @return {number}
 */
 var trap = function(height) {
    if(height.length < 3) return 0
    let leftMax = 0, rightMax = 0
    let left = 0, right = height.length - 1
    let sum = 0

    while(left <= right) {
        leftMax = Math.max(leftMax, height[left])
        rightMax = Math.max(rightMax, height[right])

        if(leftMax <= rightMax) {
            sum += (leftMax-height[left])
            left++
        } else {
            sum += (rightMax-height[right])
            right--
        }
    }

    return sum
};

/**
 * 84. 柱状图中最大的矩形 https://leetcode.cn/problems/largest-rectangle-in-histogram/description/?favorite=2cktkvj
 * 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
   求在该柱状图中，能够勾勒出来的矩形的最大面积。
 */

/**
 * @param {number[]} heights
 * @return {number}
 */
 var largestRectangleArea = function(heights) {
    let area = 0
    const stack = [0]
    heights = [0, ...heights, 0]
    for(let i=1; i<heights.length; i++) {
        while(heights[i] < heights[stack[stack.length-1]]) {
            const lastTopIndex = stack.pop()
            area = Math.max(area,
                heights[lastTopIndex] * (i - stack[stack.length-1] - 1)
            )
        }
        stack.push(i)
    }

    return area
};


/*******************双指针 end******************************/

/**
 * 347. 前 K 个高频元素 https://leetcode.cn/problems/top-k-frequent-elements/description/?favorite=2cktkvj
 * 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
 */

/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number[]}
 */
 var topKFrequent = function(nums, k) {
    if(nums.length == k) return nums
    const map = new Map()
    const arr = [...new Set(nums)];

    nums.forEach(num => {
        if(map.get(num)) {
            map.set(num, map.get(num) + 1)
        } else {
            map.set(num, 1)
        }
    })

    return arr.sort((a, b) => map.get(b)-map.get(a)).slice(0,k)
};

/**
 * 394. 字符串解码 https://leetcode.cn/problems/decode-string/description/?favorite=2cktkvj
 * 给定一个经过编码的字符串，返回它解码后的字符串。
 */

/**
 * @param {string} s
 * @return {string}
 */
 var decodeString = function(s) {
    const num_arr = []
    const str_arr = []
    let result = ''
    let count = 0
    for(const str of s) {
        if(!isNaN(str)) {
            count = count*10 + Number(str)
        } else if(str == '['){
            str_arr.push(result)
            result = ''
            num_arr.push(count)
            count = 0
        } else if(str == ']') {
            const repeatCount = num_arr.pop()
            result = str_arr.pop() + result.repeat(repeatCount)
        } else {
            result += str
        }
    }
    return result 
};

/**
 * 438. 找到字符串中所有字母异位词 https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?favorite=2cktkvj
 * 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
    异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
 */

/**
 * @param {string} s
 * @param {string} p
 * @return {number[]}
 */
 var findAnagrams = function(s, p) {
    const pn = p.length, sn = s.length
    if(sn < pn) return []
    const result = []
    const s_recode = new Array(26).fill(0)
    const p_recode = new Array(26).fill(0)
    for(let i=0; i<pn; i++) {
        ++s_recode[s[i].charCodeAt()-'a'.charCodeAt()];
        ++p_recode[p[i].charCodeAt()-'a'.charCodeAt()]
    }
    if(s_recode.toString() === p_recode.toString()) {
        result.push(0)
    }
    for(let i=0; i<sn-pn; i++) {
        --s_recode[s[i].charCodeAt()-'a'.charCodeAt()];
        ++s_recode[s[i+pn].charCodeAt()-'a'.charCodeAt()];
        if(s_recode.toString() === p_recode.toString()) {
            result.push(i+1)
        }
    }
    return result
};

/**
 * 448. 找到所有数组中消失的数字 https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/description/?favorite=2cktkvj
 * 给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，并以数组的形式返回结果。
 */

/**
 * @param {number[]} nums
 * @return {number[]}
 */
 var findDisappearedNumbers = function(nums) {
    const n = nums.length
    for(const num of nums) {
        let x = (num-1) % n
        nums[x] += n
    }
    const result = []
    for (let i=0; i<n; i++) {
        if(nums[i]<=n) {
            result.push(i+1)
        }
    }
    return result
};

/**
 * 461. 汉明距离 https://leetcode.cn/problems/hamming-distance/description/?favorite=2cktkvj
 * 两个整数之间的 汉明距离 指的是这两个数字对应二进制位不同的位置的数目。
 */

/**
 * @param {number} x
 * @param {number} y
 * @return {number}
 */
 var hammingDistance = function(x, y) {
    let s = x ^ y, ret = 0;
    while (s != 0) {
        ret += s & 1;
        s >>= 1;
    }
    return ret;
};

/**
 * 560. 和为 K 的子数组 https://leetcode.cn/problems/subarray-sum-equals-k/description/?favorite=2cktkvj
 * 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的连续子数组的个数 。
 */

/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
 */
 var subarraySum = function(nums, k) {
    const map = {0:1}
    let count = 0, prefix = 0
    for(let i = 0; i<nums.length; i++) {
        prefix += nums[i]

        if(map[prefix-k]) {
            count += map[prefix-k]
        }

        if(map[prefix]) {
            map[prefix]++
        } else {
            map[prefix] = 1
        }
    }
    return count
};

/**
 * 581. 最短无序连续子数组 https://leetcode.cn/problems/shortest-unsorted-continuous-subarray/description/?favorite=2cktkvj
 * 给你一个整数数组 nums ，你需要找出一个 连续子数组 ，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
    请你找出符合题意的 最短 子数组，并输出它的长度。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var findUnsortedSubarray = function(nums) { // 双指针，right：值比前面最大值小的 left: 值比后面最小值小的
    const n = nums.length;
    let min = Number.MAX_SAFE_INTEGER
    let left = -1
    let max = Number.MIN_SAFE_INTEGER
    let right = -1

    for(let i=0; i<nums.length; i++) {
        if(max > nums[i]) {
            right = i
        } else {
            max = nums[i]
        }

        if(min < nums[n-i-1]) {
            left = n-i-1
        } else {
            min = nums[n-i-1]
        }
    }

    return right === -1 ? 0 : right-left+1
};

/**
 * 621. 任务调度器 https://leetcode.cn/problems/task-scheduler/description/?favorite=2cktkvj
 * 给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。
 * 在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。
    然而，两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。
    你需要计算完成所有任务所需要的 最短时间 。
 */

/**
 * @param {character[]} tasks
 * @param {number} n
 * @return {number}
 */
 var leastInterval = function(tasks, n) {
    let maxCount = 0
    const map = {}
    for(let i=0; i<tasks.length; i++) {
        if(map[tasks[i]]) {
            map[tasks[i]]++;
        } else {
            map[tasks[i]] = 1
        }
        maxCount = Math.max(maxCount, map[tasks[i]])
    }

    let ans = (maxCount-1) * (n+1)
    const values = Object.values(map)
    values.forEach(count => {
        if(count === maxCount) {
            ans++
        }
    })

    return Math.max(ans, tasks.length)
};

/**
 * 739. 每日温度 https://leetcode.cn/problems/daily-temperatures/description/?favorite=2cktkvj
 * 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，
 * 下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。
 */

/**
 * @param {number[]} temperatures
 * @return {number[]}
 */
 var dailyTemperatures = function(temperatures) { // 暴力
    const T = temperatures
    const res = new Array(T.length).fill(0);
    for (let i = 0; i < T.length; i++) {
        for (let j = i + 1; j < T.length; j++) {
            if (T[j] > T[i]) {
                res[i] = j - i;
                break;
            }
        }
    }
    return res;
};

var dailyTemperatures = function(temperatures) { // 单调栈
    const n = temperatures.length
    const ans = new Array(n).fill(0)

    const stack = []

    for(let i=n-1; i>=0; i--) {
        while(stack.length && temperatures[stack[stack.length-1]] <= temperatures[i]) {
            stack.pop()
        }

        if(stack.length) {
            ans[i] = stack[stack.length-1] - i
        }

        stack.push(i)
    }

    return ans
};

/**
 * 189. 轮转数组 https://leetcode.cn/problems/rotate-array/description/
 * 给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
 * 输入: nums = [1,2,3,4,5,6,7], k = 3
   输出: [5,6,7,1,2,3,4]
 */
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {void} Do not return anything, modify nums in-place instead.
 */
 var rotate = function(nums, k) { // 翻转法
    k %= nums.length;
    reverse(nums, 0, nums.length - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, nums.length - 1);
    return nums;
};
let reverse = function(nums, start, end){
    while(start < end){
        [nums[start++], nums[end--]] = [nums[end], nums[start]];
    }
}

var rotate = function(nums, k) { // 截断拼接法
    const n = nums.length
    k = k % n
    const headArr = nums.splice(n-k, k)
    nums.unshift(...headArr)
};

/**
 * 73. 矩阵置零 https://leetcode.cn/problems/set-matrix-zeroes/description/
 * 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
 */

/**
 * @param {number[][]} matrix
 * @return {void} Do not return anything, modify matrix in-place instead.
 */
 var setZeroes = function(matrix) { // 思路：用matrix的第一行，第一列记录是否为0标识位
    let m = matrix.length, n=matrix[0].length;
    let colZero = false

    for(let i=0; i<m; i++) {
        if(matrix[i][0]===0) {
            colZero = true
        } 

        for(let j=1; j<n; j++) {
            if(matrix[i][j]===0) {
                matrix[i][0] = matrix[0][j] = 0
            }
        }
    }

    for(let i=m-1; i>=0; i--) {
        for(let j=1; j<n; j++) {
            if(matrix[0][j]===0 || matrix[i][0]===0) {
                matrix[i][j]=0
            }
        }
        if(colZero) {
            matrix[i][0] = 0
        }
    }
};

/**
 * 54. 螺旋矩阵 https://leetcode.cn/problems/spiral-matrix/description/
 * 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
 */

/**
 * @param {number[][]} matrix
 * @return {number[]}
 */
 var spiralOrder = function(matrix) {
    if(!matrix.length || !matrix[0].length) return []
    const m=matrix.length, n=matrix[0].length
    let l=0, r=n-1, top=0, bottom=m-1
    const result=[]
    while(l<=r && top<=bottom) {
        for(let i=l; i<=r; i++) {
            result.push(matrix[top][i])
        }
        for(let j=top+1; j<=bottom; j++) {
            result.push(matrix[j][r])
        }
        if(l<r && top<bottom) {
            for(let i=r-1; i>l; i--) {
                result.push(matrix[bottom][i])
            }
            for(let j=bottom; j>top; j--) {
                result.push(matrix[j][l])
            }
        }
        [l, r, top, bottom] = [l+1, r-1, top+1, bottom-1]
    }
    return result
};

/**
 * 239. 滑动窗口最大值 https://leetcode.cn/problems/sliding-window-maximum/description/
 * 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
   返回 滑动窗口中的最大值 。
   输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
   输出：[3,3,5,5,6,7]
 */

/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number[]}
 */
var maxSlidingWindow = function(nums, k) { // 单调队列
    const ans = []
    const q = []
    for(let i=0; i<nums.length; i++) {
        while(q.length && nums[q[q.length-1]]<=nums[i]) {
            q.pop()
        }
        q.push(i)
        if(i-q[0]>=k) {
            q.shift()
        }
        if(i>=k-1) {
            ans.push(nums[q[0]])
        }
    }
    return ans
};

/**
 * 295. 数据流的中位数 https://leetcode.cn/problems/find-median-from-data-stream/description/
    中位数是有序整数列表中的中间值。
    例如 arr = [2,3,4] 的中位数是 3 。
    例如 arr = [2,3] 的中位数是 (2 + 3) / 2 = 2.5 。

    实现 MedianFinder 类:

    MedianFinder() 初始化 MedianFinder 对象。

    void addNum(int num) 将数据流中的整数 num 添加到数据结构中。

    double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。
 */

var MedianFinder = function() {
    this.data = []
};

/** 
 * @param {number} num
 * @return {void}
 */
MedianFinder.prototype.addNum = function(num) {
    const n = this.data.length
    if(!n) {
        this.data.push(num)
        return
    }

    let l=0, r=n-1
    while(l<=r) {
        const mid = l + Math.floor((r-l)/2)
        if(this.data[mid] > num) {
            r = mid - 1
        } else {
            l = mid + 1
        }
    }
    this.data.splice(r+1, 0, num)
};

/**
 * @return {number}
 */
MedianFinder.prototype.findMedian = function() {
    const n = this.data.length
    if(!n) return null
    if(n%2) {
        return this.data[Math.floor(n/2)]
    } else {
        const midleft = Math.floor((n-1)/2)
        return (this.data[midleft] + this.data[midleft+1])/2
    }
};

/**
 * 41. 缺失的第一个正数 https://leetcode.cn/problems/first-missing-positive/description/
 * 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
 * 输入：nums = [3,4,-1,1]
   输出：2
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var firstMissingPositive = function(nums) {
    const n = nums.length
    for(let i=0; i<n; i++) {
        while(nums[i]>=1 && nums[i]<=n && nums[nums[i]-1]!=nums[i]) {
            const temp = nums[nums[i]-1]
            nums[nums[i]-1] = nums[i]
            nums[i] = temp
        }
    }
    for(let i=0; i<n; i++) {
        if(nums[i]!=i+1) {
            return i+1
        }
    }
    return n+1
};