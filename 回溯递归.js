/**
 * 494. 目标和 https://leetcode.cn/problems/target-sum/description/?favorite=2cktkvj
 * 给你一个整数数组 nums 和一个整数 target 。
    向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
    例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
    返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
 */

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
 var findTargetSumWays = function(nums, target) {
    let count = 0

    function dfs(index, sum) {
        if(index === nums.length) {
            if(sum === target) {
                count++
            }
            return 
        } 

        dfs(index+1, sum+nums[index]);
        dfs(index+1, sum-nums[index]);
    }

    dfs(0, 0)

    return count
};

