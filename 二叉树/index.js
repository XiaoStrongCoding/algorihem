/**
 * 二叉树：分为两类：普通和二叉搜索树(left<mid<right)
 * 经典题型：深度遍历dfs(deep first search) 和广度遍历
 * 本质：上述便利和递归的结合
 * 注意：
 *     叶子节点：必须判断左右子节点都没有
 */

/**
 * 94. 二叉树的中序遍历 https://leetcode.cn/problems/binary-tree-inorder-traversal/
 * 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
 */

/**
 * @param {TreeNode} root
 * @return {number[]}
 */
 var inorderTraversal = function(root) {
    const result = []

    function dfs(node) {
        if(!node) return null

        dfs(node.left);
        result.push(node.val);
        dfs(node.right)
    }

    dfs(root)
    return result
};

/**
 * 102. 二叉树的层序遍历 https://leetcode.cn/problems/binary-tree-level-order-traversal/
 * 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
 */

/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
 var levelOrder = function(root) {
    const res = []
    if(!root) return res
    const queue = [root]
    while(queue.length) {
        res.push([]);
        const lastIndex = res.length - 1;
        const n = queue.length
        for(let i=0; i<n; i++) {
            const node = queue.shift()
            res[lastIndex].push(node.val)

            node.left && queue.push(node.left);
            node.right && queue.push(node.right);
        }
    }

    return res
};

/**
 * 101. 对称二叉树 https://leetcode.cn/problems/symmetric-tree/
 * 给你一个二叉树的根节点 root ， 检查它是否轴对称
 */

/**
 * @param {TreeNode} root
 * @return {boolean}
 */
var isSymmetric = function(root) {
    function check(left, right) {
        if(!left && !right) return true
        if(!left || !right) return false
        return  left.val === right.val && check(left.left, right.right) && check(left.right, right.left)
    }

    return check(root.left, root.right)
};

/**
 * 104. 二叉树的最大深度 https://leetcode.cn/problems/maximum-depth-of-binary-tree/
 * 给定一个二叉树，找出其最大深度。
    二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
    说明: 叶子节点是指没有子节点的节点。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
 var maxDepth = function(root) {
    if(!root) {
        return 0
    } else {
        const l = maxDepth(root.left)
        const r = maxDepth(root.right)
        return Math.max(l, r) + 1
    }
};

/**
 * 111. 二叉树的最小深度 https://leetcode.cn/problems/minimum-depth-of-binary-tree/
 * 给定一个二叉树，找出其最小深度。
    最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
    说明：叶子节点是指没有子节点的节点。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
var minDepth = function(root) {
    if(!root) {
        return 0
    } else if(!root.right && !root.left) {
        return 1
    } else if(!root.right) {
        return 1 + minDepth(root.left)
    } else if(!root.left) {
        return 1 + minDepth(root.right)
    } else {
        const l = minDepth(root.left)
        const r = minDepth(root.right)
        return Math.min(l, r) + 1
    }
};

/**
 * 222. 完全二叉树的节点个数 https://leetcode.cn/problems/count-complete-tree-nodes/
 * 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
    完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，
    并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
var countNodes = function(root) {
    if(!root) {
        return 0
    } else {
        return countNodes(root.left) + countNodes(root.right) + 1
    }
};

/**
 * 110. 平衡二叉树 https://leetcode.cn/problems/balanced-binary-tree/
 * 给定一个二叉树，判断它是否是高度平衡的二叉树。
    本题中，一棵高度平衡二叉树定义为：
    一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 
 */

/**
 * @param {TreeNode} root
 * @return {boolean}
 */
 var isBalanced = function(root) {

    function getDeps(node) {
        if(!node) {
            return 0
        } else {
            const l = getDeps(node.left)
            if(l === -1 ) {
                return -1
            }
            const r = getDeps(node.right)
            if(r === -1 ) {
                return -1
            }

            if(Math.abs(l-r) > 1) {
                return -1
            } else {
                return 1 + Math.max(l, r)
            }
        }
    }
    
    return getDeps(root) !== -1
};

/**
 * 257. 二叉树的所有路径 https://leetcode.cn/problems/binary-tree-paths/
 * 给你一个二叉树的根节点 root ，按 任意顺序 ，返回所有从根节点到叶子节点的路径。
    叶子节点 是指没有子节点的节点。
 */

/**
 * @param {TreeNode} root
 * @return {string[]}
 */
 var binaryTreePaths = function(root) {
    const result = []

    function buildPath(node, path) {
        if(!node) return ''

        path += node.val
        if(!node.left && !node.right) {
            result.push(path)
            return
        }

        path += '->';
        node.left && buildPath(node.left, path);
        node.right && buildPath(node.right, path);
    }

    buildPath(root, '');

    return result
};

/**
 * 404. 左叶子之和 https://leetcode.cn/problems/sum-of-left-leaves/
 * 给定二叉树的根节点 root ，返回所有左叶子之和。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
 var sumOfLeftLeaves = function(root) { // 需加深
    let result = 0

    function findLeft(node) {
        if(!node) return

        if(node.left && !node.left.left && !node.left.right) {
            result += node.left.val;
        }
        findLeft(node.left);
        findLeft(node.right)
    }

    findLeft(root)

    return result
};

/**
 * 513. 找树左下角的值 https://leetcode.cn/problems/find-bottom-left-tree-value/
 * 给定一个二叉树的 根节点 root，请找出该二叉树的 最底层 最左边 节点的值。
    假设二叉树中至少有一个节点。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
 var findBottomLeftValue = function(root) { // 题意容易理解错
    let result = root.val
    let height = 0

    function findLeft(node, layer) {
        if(!node) return

        layer++;
        if(layer > height) {
            height = layer;
            result = node.val;
        }

        findLeft(node.left, layer);
        findLeft(node.right, layer);
    }

    findLeft(root, 0);

    return result
};

/**
 * 112. 路径总和
 * 给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，
 * 这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。
   叶子节点 是指没有子节点的节点。
 */

/**
 * @param {TreeNode} root
 * @param {number} targetSum
 * @return {boolean}
 */
var hasPathSum = function(root, targetSum) {
    let result = false
    function addPath(node, sum) {
        if(!node) return

        sum += node.val;
        if(!node.left && !node.right &&  sum === targetSum) {
            result = true;
            return
        }

        node.left && addPath(node.left, sum);
        node.right && addPath(node.right, sum);
    }

    addPath(root, 0)

    return result
};

/**
 * 226. 翻转二叉树 https://leetcode.cn/problems/invert-binary-tree/
 * 给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
 */

/**
 * @param {TreeNode} root
 * @return {TreeNode}
 */
 var invertTree = function(root) {
    function invert(node) {
        if(!node) return

        const temp = node.left;
        node.left = node.right;
        node.right = temp;

        invert(node.left);
        invert(node.right);
    }

    invert(root)

    return root
};

/**
 * 617. 合并二叉树 https://leetcode.cn/problems/merge-two-binary-trees/
 * 给你两棵二叉树： root1 和 root2 。
   想象一下，当你将其中一棵覆盖到另一棵之上时，两棵树上的一些节点将会重叠（而另一些不会）。
   你需要将这两棵树合并成一棵新二叉树。合并的规则是：如果两个节点重叠，那么将这两个节点的值相加作为合并后节点的新值；
   否则，不为 null 的节点将直接作为新二叉树的节点。
   返回合并后的二叉树。
   注意: 合并过程必须从两个树的根节点开始。
 */

/**
 * @param {TreeNode} root1
 * @param {TreeNode} root2
 * @return {TreeNode}
 */
 var mergeTrees = function(t1, t2) {
    if (t1 == null && t2) {
        return t2;
    }
    if ((t1 && t2 == null) || (t1 == null && t2 == null)) {
        return t1;
    }
    t1.val += t2.val;

    t1.left = mergeTrees(t1.left, t2.left);
    t1.right = mergeTrees(t1.right, t2.right);

    return t1;
};

/**
 * 654. 最大二叉树 https://leetcode.cn/problems/maximum-binary-tree/description/
 * 给定一个不重复的整数数组 nums 。 最大二叉树 可以用下面的算法从 nums 递归地构建:
    创建一个根节点，其值为 nums 中的最大值。
    递归地在最大值 左边 的 子数组前缀上 构建左子树。
    递归地在最大值 右边 的 子数组后缀上 构建右子树。
    返回 nums 构建的 最大二叉树 。
 */

/**
 * @param {number[]} nums
 * @return {TreeNode}
 */
 var constructMaximumBinaryTree = function(nums) {

    function buildTree(left, right) {
        if(left>right) return null

        let max = -Infinity
        let mid = 0
        for(let i=left; i<=right; i++) {
            if(nums[i] > max) {
                max = nums[i];
                mid = i
            }
        }

        const root = new TreeNode(max);
        root.left = buildTree(left, mid - 1);
        root.right = buildTree(mid+1, right)

        return root
    }

    return buildTree(0, nums.length-1)
};

/**
 * 114. 二叉树展开为链表 https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/?favorite=2cktkvj
 * 给你二叉树的根结点 root ，请你将它展开为一个单链表：
    展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
    展开后的单链表应该与二叉树 先序遍历 顺序相同。
 */

/**
 * @param {TreeNode} root
 * @return {void} Do not return anything, modify root in-place instead.
 */
var flatten = function(root) {
    let cur = root
    while(cur) {
        if(cur.left) {
            const temp = cur.left
            let rightNode = cur.left
            while(rightNode.right) {
                rightNode = rightNode.right
            }
            rightNode.right = cur.right
            cur.left = null
            cur.right = temp 
        }
        cur = cur.right
    }
};

/**
 * 124. 二叉树中的最大路径和 https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/?favorite=2cktkvj
 * 路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。
 * 同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
    路径和 是路径中各节点值的总和。
    给你一个二叉树的根节点 root ，返回其 最大路径和 。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
 var maxPathSum = function(root) {
    let maxSum = Number.MIN_SAFE_INTEGER

    function dfs(node) {
        if(node == null) return 0

        const left = dfs(node.left)
        const right = dfs(node.right)

        maxSum = Math.max(maxSum, left + right + node.val)
        const output = node.val + Math.max(0, left, right)
        return output>0 ? output : 0
    }

    dfs(root)
    return maxSum
};

/**
 * 437. 路径总和 III https://leetcode.cn/problems/path-sum-iii/description/
 * 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
   路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
 */

/**
 * @param {TreeNode} root
 * @param {number} targetSum
 * @return {number}
 */
 var pathSum = function(root, targetSum) {
    if(!root) return 0

    let count = rootSum(root, targetSum)
    count+=pathSum(root.left, targetSum)
    count+=pathSum(root.right, targetSum)

    return count
};

function rootSum(node, target) {
    if(!node) return 0
    let ret = 0
    if(node.val === target) {
        ret++
    }
    ret+=rootSum(node.left, target-node.val)
    ret+=rootSum(node.right, target-node.val)
    return ret
}


/**
 * 543. 二叉树的直径 https://leetcode.cn/problems/diameter-of-binary-tree/description/
 * 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
 var diameterOfBinaryTree = function(root) {
    let count = 0

    function dfs(node) {
        if(!node) return 0

        const l = dfs(node.left)
        const r = dfs(node.right)
        
        count = Math.max(count, l+r)
        return Math.max(l,r) + 1
    }

    dfs(root)

    return count
};

//************************************二叉搜索树 start********************************

/**
 * 700. 二叉搜索树中的搜索 https://leetcode.cn/problems/search-in-a-binary-search-tree/
 * 给定二叉搜索树（BST）的根节点 root 和一个整数值 val。
   你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 
   如果节点不存在，则返回 null 。
 */

/**
 * @param {TreeNode} root
 * @param {number} val
 * @return {TreeNode}
 */
 var searchBST = function(root, val) {
    if(!root) return null

    if(root.val === val) return root
    if(root.val > val) {
        return searchBST(root.left, val)
    } else {
       return searchBST(root.right, val)
    }
};

/**
 * 98. 验证二叉搜索树 https://leetcode.cn/problems/validate-binary-search-tree/
 * 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
    有效二叉搜索树定义如下：
        节点的左子树只包含 小于 当前节点的数。
        节点的右子树只包含 大于 当前节点的数。
        所有左子树和右子树自身必须也是二叉搜索树。
 */

/**
 * @param {TreeNode} root
 * @return {boolean}
 */
 var isValidBST = function(root) {
    function dfs(node, low, high) {
        if(!node) return true // 容易忘
        if(node.val <= low || node.val >= high) return false
        return dfs(node.left, low, node.val) && dfs(node.right, node.val, high)
    }
    return dfs(root, -Infinity, Infinity)
};

/**
 * 530. 二叉搜索树的最小绝对差 https://leetcode.cn/problems/minimum-absolute-difference-in-bst/
 * 给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。
   差值是一个正数，其数值等于两值之差的绝对值。
 */

/**
 * @param {TreeNode} root
 * @return {number}
 */
var getMinimumDifference = function(root) {
    let min = Infinity

    function dfs(node, low, high) {
        if(!node) return

        const value = node.val
        min = Math.min(min, Math.abs(value - low), Math.abs(value - high));

        dfs(node.left, low, value);
        dfs(node.right, value, high);
    }

    dfs(root, Infinity, Infinity);

    return min
};

/**
 * 501. 二叉搜索树中的众数 https://leetcode.cn/problems/find-mode-in-binary-search-tree/
 * 给你一个含重复值的二叉搜索树（BST）的根节点 root ，找出并返回 BST 中的所有 众数
 * （即，出现频率最高的元素）。
    如果树中有不止一个众数，可以按 任意顺序 返回。
    假定 BST 满足如下定义：
        结点左子树中所含节点的值 小于等于 当前节点的值
        结点右子树中所含节点的值 大于等于 当前节点的值
        左子树和右子树都是二叉搜索树
 */

/**
 * @param {TreeNode} root
 * @return {number[]}
 */
 var findMode = function(root) {
    let base = 0, count = 0, maxCount = 0
    let result = []

    function dfs(node) {
        if (!node) return 

        dfs(node.left);

        if(base === node.val) {
            count++;
        } else {
            base = node.val;
            count = 1
        }

        if(count === maxCount) {
            result.push(base)
        }

        if(count > maxCount) {
            maxCount = count
            result = [base]
        }

        dfs(node.right)
    }

    dfs(root)

    return result
};

/**
 * 538. 把二叉搜索树转换为累加树 https://leetcode.cn/problems/convert-bst-to-greater-tree/
 * 给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），
 * 使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
    提醒一下，二叉搜索树满足下列约束条件：
        节点的左子树仅包含键 小于 节点键的节点。
        节点的右子树仅包含键 大于 节点键的节点。
        左右子树也必须是二叉搜索树。
 */

/**
 * @param {TreeNode} root
 * @return {TreeNode}
 */
 var convertBST = function(root) {
    let sum = 0
    function dfs(node) {
        if(!node) return
        dfs(node.right)
        sum += node.val;
        node.val = sum;
        dfs(node.left);
    }

    dfs(root);

    return root
};

/**
 * 108. 将有序数组转换为二叉搜索树 https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/
 * 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
   高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
 */

/**
 * @param {number[]} nums
 * @return {TreeNode}
 */
 var sortedArrayToBST = function(nums) {
    function buildBST(nums, start, end) {
        if(start>end) return null

        const mid = (start+end) >> 1

        const root = new TreeNode(nums[mid])

        root.left = buildBST(nums, start, mid - 1);
        root.right = buildBST(nums, mid+1, end);

        return root;
    }

    return buildBST(nums, 0, nums.length - 1)
};

/**
 * 701. 二叉搜索树中的插入操作 https://leetcode.cn/problems/insert-into-a-binary-search-tree/
 * 
 */

/**
 * @param {TreeNode} root
 * @param {number} val
 * @return {TreeNode}
 */
var insertIntoBST = function(root, val) {
    if(!root) {
        return new TreeNode(val)
    }

    if(root.val > val) {
        root.left = insertIntoBST(root.left, val)
        return root
    }

    if(root.val < val) {
        root.right = insertIntoBST(root.right, val)
        return root
    }
};

/**
 * 450. 删除二叉搜索树中的节点 https://leetcode.cn/problems/delete-node-in-a-bst/
 * 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。
 * 返回二叉搜索树（有可能被更新）的根节点的引用。
    一般来说，删除节点可分为两个步骤：
        首先找到需要删除的节点；
        如果找到了，删除它。
 */

/**
 * @param {TreeNode} root
 * @param {number} key
 * @return {TreeNode}
 */
 var deleteNode = function(root, key) {
    if(!root) return null

    if(root.val > key) {
        root.left = deleteNode(root.left, key)
        return root
    }

    if(root.val < key) {
        root.right = deleteNode(root.right, key)
        return root
    }

    if(root.val === key) {
        if(!root.left && !root.right) return null

        if(!root.left) return root.right
        if(!root.right) return root.left

        let target = root.right
        while(target.left) {
            target = target.left
        }
        root.right = deleteNode(root.right, target.val);
        target.left = root.left;
        target.right = root.right;

        return target
    }
};

/**
 * 669. 修剪二叉搜索树 https://leetcode.cn/problems/trim-a-binary-search-tree/
 * 给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。通过修剪二叉搜索树，
 * 使得所有节点的值在[low, high]中。
 */

/**
 * @param {TreeNode} root
 * @param {number} low
 * @param {number} high
 * @return {TreeNode}
 */
 var trimBST = function(root, low, high) {
    if(!root) return null

    if(root.val < low) {
        return trimBST(root.right, low, high)
    } 

    if(root.val > high) {
        return trimBST(root.left, low, high)
    }

    root.left = trimBST(root.left, low, high);
    root.right = trimBST(root.right, low, high)

    return root
};

/**
 * 235. 二叉搜索树的最近公共祖先 https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/
 * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
 */

/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
 var lowestCommonAncestor = function(root, p, q) {
    if (p.val < root.val && q.val < root.val) {
        return lowestCommonAncestor(root.left, p, q);
    }
    if (p.val > root.val && q.val > root.val) {
        return lowestCommonAncestor(root.right, p, q);
    }
    return root;
};

/**
 * 236. 二叉树的最近公共祖先 https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/
 * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
 */

/**
 * @param {TreeNode} root
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {TreeNode}
 */
 var lowestCommonAncestor = function(root, p, q) {
    let ans;

    function dfs(node, p, q) {
        if(!node) return false

        const lc = dfs(node.left, p ,q);
        const rc = dfs(node.right, p ,q);

        if(ans) return

        if((lc&&rc) || (node.val===p.val||node.val===q.val)&&(lc||rc)) {
            ans = node
        }

        return lc || rc || (node.val===p.val||node.val===q.val)
    }

    dfs(root, p, q)

    return ans
};

//************************************二叉搜索树 end********************************