/**
 * 二分法 O(logn)
 * 经典：旋转单调数组
 * 特殊变型 例：4. 寻找两个正序数组的中位数 没太理解
 * 
 * 使用场景，其实比较受限，最明显的特点是：
    绝大情况，查找目标具有单调性质（单调递增、单调递减）
    有上下边界，并且最好能够通过index下标访问元素
 */

/**
 * 35. 搜索插入位置 https://leetcode.cn/problems/search-insert-position/description/
 * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
 */
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
 var searchInsert = function(nums, target) {
    const n = nums.length
    let l=0, r = n-1
    while(l<=r) {
        const mid = l + Math.floor((r-l)/2)
        if(nums[mid]===target) {
            return mid
        } else if(nums[mid]>target) {
            r = mid-1
        } else {
            l = mid + 1
        }
    }
    return l
};


/**
 * 34. 在排序数组中查找元素的第一个和最后一个位置 https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?favorite=2cktkvj
 * 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
 */

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
 var searchRange = function(nums, target) {
    if(!nums.length) return [-1, -1]

    let left = 0, right = nums.length - 1

    while(left <= right) {
        let mid = left + ((right-left)>>1)

        if(nums[mid] === target) {
            let l = mid - 1, r = mid + 1
            while(l>=0 && nums[l] === target) l--;
            while(r<nums.length && nums[r] === target) r++;
            return [l+1, r-1]
        }else if(nums[mid] > target) {
            right = mid - 1
        }else {
            left = mid + 1
        }
    }

    return [-1, -1]
};

/**
 * 69. x 的平方根 https://leetcode.cn/problems/sqrtx/
 * 给你一个非负整数 x ，计算并返回 x 的 算术平方根 。
 */

 const mySqrt = function(x) {
    if (x < 2) return x
    let left = 1, mid, right = Math.floor(x / 2);
    while (left <= right) {
       mid = Math.floor(left + (right - left) / 2)
       if (mid * mid === x) return mid
       if (mid * mid < x) {
           left = mid + 1
       }else {
           right = mid - 1
       }
    }
    return right
}

/**
 * 153. 寻找旋转排序数组中的最小值 https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/
 */

/**
 * @param {number[]} nums
 * @return {number}
 */
 var findMin = function(nums) {
    if(!nums.length) return null
    if(nums.length === 1) return nums[0]
    let left = 0, right = nums.length - 1, mid
    // 此时数组单调递增，first element就是最小值
    if (nums[right] > nums[left]) return nums[0]
    while (left <= right) {
        mid = left + ((right - left) >> 1)
        if (nums[mid] > nums[mid + 1]) {
            return nums[mid + 1]
        }
        if (nums[mid] < nums[mid - 1]) {
            return nums[mid]
        }
        if (nums[mid] > nums[0]) {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return null
};

/**
 * 33. 搜索旋转排序数组 https://leetcode.cn/problems/search-in-rotated-sorted-array/
 * 整数数组 nums 按升序排列，数组中的值 互不相同 。
 */

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
 var search = function(nums, target) {
    if (!nums.length) return -1
    let left = 0, right = nums.length - 1, mid
    while (left <= right) {
        mid = left + ((right - left) >> 1)
        if (nums[mid] === target) {
            return mid
        }
        if (nums[mid] >= nums[left]) {
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    return -1
};

/**
 * 81. 搜索旋转排序数组 II https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/
 * 已知存在一个按非降序排列的整数数组 nums ，数组中的值可以相同。
 */

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {boolean}
 */
 var search = function(nums, target) {
    if (!nums.length) return false
    let left = 0, right = nums.length - 1, mid
    while (left <= right) {
        mid = left + ((right - left) >> 1)
        if (nums[mid] === target) {
            return true
        }
        if (nums[left] === nums[mid]) {
            ++left
            continue
        }
        if (nums[mid] >= nums[left]) {
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    return false
};

/** 特殊变形
 * 4. 寻找两个正序数组的中位数 https://leetcode.cn/problems/median-of-two-sorted-arrays/description/
 * 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
    算法的时间复杂度应该为 O(log (m+n)) 。
 */

/**
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number}
 */
 var findMedianSortedArrays = function(nums1, nums2) {
    if(nums1.length > nums2.length) {
        [nums1, nums2] = [nums2, nums1]
    }

    const n1 = nums1.length
    const n2 = nums2.length

    const leftCounts = Math.floor((n1+n2+1)/2)
    let l = 0; r = n1;
    while(l<r) {
        const i = l + Math.floor((r-l+1)/2)
        const j = leftCounts - i

        if(nums1[i-1] > nums2[j]) {
            r = i-1
        } else {
            l = i
        }
    }

    const j = leftCounts - l

    const maxLeft1 = l === 0 ? -Infinity : nums1[l-1];
    const maxLeft2 = j === 0 ? -Infinity : nums2[j-1];
    const minRight1 = l === n1 ? Infinity : nums1[l];
    const minRight2 = j === n2 ? Infinity : nums2[j];

    if((n1+n2)%2===1) {
        return Math.max(maxLeft1, maxLeft2)
    } else {
        return (Math.max(maxLeft1, maxLeft2)+Math.min(minRight1, minRight2))/2
    }
};

/**
 * 153. 寻找旋转排序数组中的最小值 https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/
 */
/**
 * @param {number[]} nums
 * @return {number}
 */
 var findMin = function(nums) {
    const n = nums.length
    let l = 0, r=n-1
    while(l<r) {
        const mid = l + Math.floor((r-l)/2)
        if(nums[mid]<nums[r]) {
            r = mid
        } else {
            l = mid + 1
        }
    }
    return nums[l]
};