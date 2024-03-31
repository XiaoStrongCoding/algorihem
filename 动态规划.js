
/**
 * 53. 最大子数组和 https://leetcode.cn/problems/maximum-subarray/description/
 * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
   子数组 是数组中的一个连续部分。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var maxSubArray = function(nums) {
    let pre = 0, maxAns = nums[0]
    nums.forEach(num => {
        pre = Math.max(pre+num, num)
        maxAns = Math.max(pre, maxAns)
    })
    return maxAns
};
//*********************贪心算法 start**************************************/

/**
 * 121. 买卖股票的最佳时机 (一次交易)https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/?favorite=2cktkvj
 * 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
    你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
    返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
 */

/**
 * @param {number[]} prices
 * @return {number}
 */
 var maxProfit = function(prices) {
    const n = prices.length
    if(n<=1) return 0
    let sell = 0
    let buy = -prices[0]
    for(let i=1; i<n; i++) {
        sell = Math.max(sell, buy+prices[i])
        buy = Math.max(-prices[i], buy)
    }
    return sell
}

/**
 * 55. 跳跃游戏 https://leetcode.cn/problems/jump-game/description/
 * 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
    数组中的每个元素代表你在该位置可以跳跃的最大长度。
    判断你是否能够到达最后一个下标。
 */

/**
 * @param {number[]} nums
 * @return {boolean}
 */
var canJump = function(nums) {
    if(nums.length===1) return true
    let cover = 0
    for(let i=0; i<=cover; i++) {
        cover = Math.max(cover, i+nums[i])
        if(cover >= nums.length-1) return true
    }
    return false
};

/**
 * 45. 跳跃游戏 II https://leetcode.cn/problems/jump-game-ii/description/
 * 给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。返回到达 nums[n - 1] 的最小跳跃次数
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var jump = function(nums) {
    let curIndex = 0, nextIndex = 0, step = 0
    for(let i=0; i<nums.length-1; i++) {
        nextIndex = Math.max(nums[i]+i, nextIndex)
        if(i=== curIndex) {
            curIndex = nextIndex
            step++
        }
    }
    return step
};

/**
 * 763. 划分字母区间 https://leetcode.cn/problems/partition-labels/description/
 * 给你一个字符串 s 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。
    注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 s 。
    返回一个表示每个字符串片段的长度的列表。
 */
/**
 * @param {string} s
 * @return {number[]}
 */
 var partitionLabels = function(s) {
    const last = new Array(26)
    const codePointA = 'a'.codePointAt()
    const n = s.length
    for(let i=0; i<n; i++) {
        last[s[i].codePointAt()-codePointA] = i
    }

    let start = 0, end=0
    const result=[]
    for(let i=0; i<n; i++) {
        end = Math.max(last[s[i].codePointAt()-codePointA], end)
        if(end === i) {
            result.push(end-start+1)
            start = end+1
        }
    }
    return result
};

//*********************贪心算法 end**************************************/

/**
 * 62. 不同路径 https://leetcode.cn/problems/unique-paths/description/
 * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
    机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
    问总共有多少条不同的路径？
 */

/**
 * @param {number} m
 * @param {number} n
 * @return {number}
 */
 var uniquePaths = function(m, n) {
    const arr = Array.from(new Array(m), () => new Array(n).fill(0))
    arr[0] = new Array(n).fill(1)
    for(let i=1; i<m; i++) {
        arr[i][0] = 1
    }

    for(let i=1; i<m; i++) {
        for(let j=1; j<n; j++) {
            arr[i][j] = arr[i-1][j] + arr[i][j-1]
        }
    }

    return arr[m-1][n-1]
};

/**
 * 64. 最小路径和 https://leetcode.cn/problems/minimum-path-sum/description/
 * 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
   说明：每次只能向下或者向右移动一步。
 */

/**
 * @param {number[][]} grid
 * @return {number}
 */
var minPathSum = function(grid) {
    const row = grid.length, column = grid[0].length
    for(let i=1; i<column; i++) {
        grid[0][i] += grid[0][i-1]
    }
    for(let i=1; i<row; i++) {
        grid[i][0] += grid[i-1][0]
    }

    for(let i=1; i<row; i++) {
        for(let j=1; j<column; j++) {
            grid[i][j] += Math.min(grid[i-1][j], grid[i][j-1])
        }
    }

    return grid[row-1][column-1]
};

/**
 * 70. 爬楼梯 https://leetcode.cn/problems/climbing-stairs/description/
 * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
   每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
 */

/**
 * @param {number} n
 * @return {number}
 */
var climbStairs = function(n) {
    if(n===1) return 1
    const dp = [1,1]
    for(let i=2; i<=n; i++) {
        dp[i] = dp[i-1] + dp[i-2]
    }
    return dp[n]
};


/**
 * 85. 最大矩形 https://leetcode.cn/problems/maximal-rectangle/description/
 * 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
 */

/**
 * @param {character[][]} matrix
 * @return {number}
 */
var maximalRectangle = function(matrix) {
    const row = matrix.length
    const col = matrix[0].length
    const dp = Array.from(new Array(row), ()=>new Array(col).fill(0))

    for(let i=0; i<row; i++) {
        for(let j=0; j<col; j++) {
            if(matrix[i][j]==0) {
                dp[i][j] = 0
            } else {
                dp[i][j] = (dp[i][j-1] || 0) + 1
            }
        }
    }

    let area = 0
    for(let i=0; i<col; i++) {
        for(let j=0; j<row; j++) {
            let height = 1
            for(let k=j+1; k<row; k++) {
                if(dp[k][i]<dp[j][i]) break
                height++
            }

            for(let k=j-1; k>=0; k--) {
                if(dp[k][i]<dp[j][i]) break
                height++
            }

            area = Math.max(area, height*dp[j][i])
        }
    }

    return area
};

/**
 * 96. 不同的二叉搜索树 https://leetcode.cn/problems/unique-binary-search-trees/?favorite=2cktkvj
 * 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。
 */

/**
 * @param {number} n
 * @return {number}
 */
var numTrees = function(n) {
    const dp = new Array(n+1).fill(0)
    dp[0] = 1
    dp[1]=1
    for(let i=2; i<=n; i++) {
        for(let j=0; j<i; j++) {
            dp[i] += (dp[j]*dp[i-1-j])
        }
    }
    return dp[n]
};

/**
 * 139. 单词拆分 https://leetcode.cn/problems/word-break/description/?favorite=2cktkvj
 * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
    注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
 */

/**
 * @param {string} s
 * @param {string[]} wordDict
 * @return {boolean}
 */
 var wordBreak = function(s, wordDict) {
    const set = new Set(wordDict)
    const len = s.length
    const dp = new Array(len+1).fill(false)
    dp[0] = true

    for(let i=1; i<=len; i++) {
        if(dp[i]) break
        for(let j=i-1; j>=0; j--) {
            if(!dp[j]) continue
            const sub = s.slice(j, i)
            if(set.has(sub) && dp[j]) {
                dp[i] = true
                break
            }
        }
    }

    return dp[len]
}; 

/**
 * 152. 乘积最大子数组 https://leetcode.cn/problems/maximum-product-subarray/description/?favorite=2cktkvj
 * 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var maxProduct = function(nums) {
    let preMin = preMax = ans = nums[0]
    
    let temp1 = temp2 = 0

    for(let i=1; i<nums.length; i++) {
        temp1 = preMin * nums[i]
        temp2 = preMax * nums[i]

        preMin = Math.min(temp1, temp2, nums[i])
        preMax = Math.max(temp1, temp2, nums[i])

        ans = Math.max(ans, preMax)
    }

    return ans
};

/**
 * 198. 打家劫舍 https://leetcode.cn/problems/house-robber/description/
 * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
 * 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
   给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var rob = function(nums) {
    const dp = [[0,0]]
    for(let i=1; i<=nums.length; i++) {
        dp[i] = [Math.max(dp[i-1][0], dp[i-1][1]), dp[i-1][0]+nums[i-1]]
    }
    return Math.max(dp[nums.length][0], dp[nums.length][1])
};


/**
 * 221. 最大正方形 https://leetcode.cn/problems/maximal-square/description/?favorite=2cktkvj
 * 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
 */

/**
 * @param {character[][]} matrix
 * @return {number}
 */
 var maximalSquare = function(matrix) {
    let maxLength = 0
    const row = matrix.length
    const column = matrix[0].length
    const dp = Array.from(new Array(row), () => new Array(column).fill(0))

    for(let i=0; i<row; i++) {
        for(let j=0; j<column; j++) {
            if(matrix[i][j] == 1) {
                if(i==0 || j==0) {
                    dp[i][j] = 1
                } else {
                    dp[i][j] = Math.min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
                }
                maxLength = Math.max(maxLength, dp[i][j])
            }
        }
    }

    return maxLength * maxLength
};

/**
 * 279. 完全平方数 https://leetcode.cn/problems/perfect-squares/description/?favorite=2cktkvj
 * 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
 */

/**
 * @param {number} n
 * @return {number}
 */
 var numSquares = function(n) {
    const dp = new Array(n+1).fill(0)

    for(let i =1; i<=n; i++) {
        dp[i] = i
        for(let j=1; i-j*j>=0; j++) {
            dp[i] = Math.min(dp[i], dp[i-j*j]+1)
        }
    }

    return dp[n]
};

/**
 * 题型 动态规划--子序列 https://leetcode.cn/problems/longest-increasing-subsequence/solutions/856903/dai-ma-sui-xiang-lu-dai-ni-xue-tou-dpzi-i1kh6/
 * 300. 最长递增子序列  https://leetcode.cn/problems/longest-increasing-subsequence/description/?favorite=2cktkvj
 * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var lengthOfLIS = function(nums) {
    const dp = new Array(nums.length).fill(1)
    let result = 1
    for(let i=1; i<nums.length; i++) {
        for(let j=0; j<i; j++) {
            if(nums[i]>nums[j]) {
                dp[i] = Math.max(dp[i], dp[j]+1)
            }
        }
        result = Math.max(result, dp[i])
    }
    return result
};

/**
 * 状态机
 * 309. 最佳买卖股票时机含冷冻期 
 * 给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。​
    设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
    卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
 */

/**
 * @param {number[]} prices
 * @return {number}
 */
 var maxProfit = function(prices) {
    const n = prices.length
    if(n===0) return n
    const hold = new Array(n)
    const unHold = new Array(n)
    hold[0] = -prices[0]
    unHold[0] = 0
    for(let i=1; i<n; i++) {
        if(i==1) {
            hold[i] = Math.max(hold[i-1], -prices[i])
        } else {
            hold[i] = Math.max(hold[i-1], unHold[i-2]-prices[i])
        }

        unHold[i] = Math.max(unHold[i-1], hold[i-1]+prices[i])
    }
    return unHold[n-1]
};

/**
 * 337. 打家劫舍 III https://leetcode.cn/problems/house-robber-iii/description/
 *  给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。
    如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
 var rob = function(root) {
    function dfs(node) {
        if(!node) return [0, 0]

        const l = dfs(node.left)
        const r = dfs(node.right)

        const select = node.val + l[1] + r[1];
        const noSelect = Math.max(l[0], l[1]) + Math.max(r[0], r[1])
        return [select, noSelect]
    }

    const res = dfs(root)
    return Math.max(res[0], res[1])
};


/**
 * 322. 零钱兑换 https://leetcode.cn/problems/coin-change/description/
 * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
    计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
    你可以认为每种硬币的数量是无限的。
 */

/**
 * @param {number[]} coins
 * @param {number} amount
 * @return {number}
 */
 var coinChange = function(coins, amount) {
    const dp = new Array(amount+1).fill(Infinity)
    dp[0] = 0
    for(let i=1; i<=amount; i++) {
        for(const coin of coins) {
            if(i-coin>=0) {
                dp[i] = Math.min(dp[i], dp[i-coin]+1)
            }
        }
    }

    return dp[amount] == Infinity ? -1 : dp[amount]
};

/**
 * 338. 比特位计数 https://leetcode.cn/problems/counting-bits/description/
 * 给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans 作为答案。
 */

/**
 * @param {number} n
 * @return {number[]}
 */
 var countBits = function(n) {
    const dp = new Array(n+1).fill(0)
    for(let i=1; i<=n; i++) {
        dp[i] = dp[i&(i-1)] + 1
    }
    return dp
};

/**
 * 背包问题 https://leetcode.cn/problems/partition-equal-subset-sum/solutions/553978/bang-ni-ba-0-1bei-bao-xue-ge-tong-tou-by-px33/
 * 416. 分割等和子集 https://leetcode.cn/problems/partition-equal-subset-sum/description/
 * 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
 */

/**
 * @param {number[]} nums
 * @return {boolean}
 */
 var canPartition = function(nums) {
    const sum = nums.reduce((sum, num)=>sum+=num, 0)
    if(sum%2) return false
    const target = sum / 2
    const dp = new Array(target+1).fill(0)
    for(let i=0; i<nums.length; i++) {
        const num = nums[i]
        for(let j=target; j>=num; j--) {
            dp[j] = Math.max(dp[j], dp[j-num]+num)
        }
    }

    return dp[target] == target
};

/**
 * 647. 回文子串 https://leetcode.cn/problems/palindromic-substrings/?favorite=2cktkvj
 * 给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
    回文字符串 是正着读和倒过来读一样的字符串。
    子字符串 是字符串中的由连续字符组成的一个序列。
    具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
 */

/**
 * @param {string} s
 * @return {number}
 */
 var countSubstrings = function(s) {

    const n = s.length
    const dp = Array.from(new Array(n), () => new Array(n).fill(false))

    let count = 0
    for(let i=0; i<n; i++) {
        for(let j=0; j<=i; j++) {
            if(s[j] === s[i]) {
                if(i-j<=1 || dp[j+1][i-1]) {
                    dp[j][i] = true
                    count++
                }
            }
        }
    }

    return count
};

/**
 * 118. 杨辉三角 https://leetcode.cn/problems/pascals-triangle/description/
 * 给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
  在「杨辉三角」中，每个数是它左上方和右上方的数的和。
 */

/**
 * @param {number} numRows
 * @return {number[][]}
 */
 var generate = function(numRows) {
    const result = []
    for(let i=0; i<numRows; i++) {
        const temp = new Array(i+1).fill(1)
        for(let j=1; j<temp.length-1; j++) {
            temp[j] = result[i-1][j-1] + result[i-1][j]
        }
        result.push(temp)
    }
    return result
};

/**
 * 1143. 最长公共子序列 https://leetcode.cn/problems/longest-common-subsequence/description/
 * 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
 * 输入：text1 = "abcde", text2 = "ace" 
  输出：3 
 */

/**
 * @param {string} text1
 * @param {string} text2
 * @return {number}
 */
 var longestCommonSubsequence = function(text1, text2) {
    const m = text1.length, n = text2.length
    const dp = Array.from(new Array(m+1), () => new Array(n+1).fill(0)) // dp[i][j]代表text1前i和text2前j的公共子序列
    for(let i=1; i<=m; i++) {
        const t1 = text1[i-1]
        for(let j=1; j<=n; j++) {
            const t2 = text2[j-1]
            if(t1===t2) {
                dp[i][j] = dp[i-1][j-1] + 1
            }else{
                dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j])
            }
        }
    }
    return dp[m][n]
};

/**
 * 72. 编辑距离 https://leetcode.cn/problems/edit-distance/description/
 * 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
    你可以对一个单词进行如下三种操作：
        插入一个字符
        删除一个字符
        替换一个字符
 */

/**
 * @param {string} word1
 * @param {string} word2
 * @return {number}
 */
 var minDistance = function(word1, word2) {
    const n1 = word1.length, n2 = word2.length
    const dp = Array.from(new Array(n1+1), ()=>new Array(n2+1).fill(0))

    for(let i=1; i<=n1; i++) {
        dp[i][0] = i
    }

    for(let i=1; i<=n2; i++) {
        dp[0][i] = i
    }

    for(let i=1; i<=n1; i++) {
        for(let j=1; j<=n2; j++) {
            if(word1[i-1]===word2[j-1]) {
                dp[i][j] = dp[i-1][j-1]
            } else {
                dp[i][j] = Math.min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
            }
        }
    }

    return dp[n1][n2]
};