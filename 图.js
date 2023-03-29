/**
 * 有向无环图
 * 解法：拓扑排序问题
    根据依赖关系，构建邻接表、入度数组。
    选取入度为 0 的数据，根据邻接表，减小依赖它的数据的入度。
    找出入度变为 0 的数据，重复第 2 步。
    直至所有数据的入度为 0，得到排序，如果还有数据的入度不为 0，说明图中存在环。
 */

/**
 * 207. 课程表 https://leetcode.cn/problems/course-schedule/description/?favorite=2cktkvj
 * 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
    在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
    例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
    请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
 */

/**
 * @param {number} numCourses
 * @param {number[][]} prerequisites
 * @return {boolean}
 */
 var canFinish = function(numCourses, prerequisites) {
    const dp = new Array(numCourses).fill(0)
    const map = {}
    for(let i=0; i<prerequisites.length; i++) {
        dp[prerequisites[i][0]]++
        if(map[prerequisites[i][1]]) {
            map[prerequisites[i][1]].push(prerequisites[i][0])
        } else {
            map[prerequisites[i][1]] = [prerequisites[i][0]]
        }
    }

    const queue = []
    for(let i=0; i<dp.length; i++) {
        if(!dp[i]) {
            queue.push(i)
        }
    }

    let count = 0
    while(queue.length) {
        const select = queue.shift();
        count++;
        const courses = map[select]
        if(courses && courses.length) {
            for(const course of courses) {
                dp[course]--
                if(!dp[course]) {
                    queue.push(course)
                }
            }
        }
    }
    return count === numCourses
};


/**
 * 399. 除法求值 https://leetcode.cn/problems/evaluate-division/description/?favorite=2cktkvj
 * 给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。
 * 每个 Ai 或 Bi 是一个表示单个变量的字符串。
    另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案
 */

/**
 * @param {string[][]} equations
 * @param {number[]} values
 * @param {string[][]} queries
 * @return {number[]}
 */
 var calcEquation = function(equations, values, queries) {
    const n = equations.length
    let count = 0
    const map = {}

    for(const [v1, v2] of equations) {
        if(!map[v1]) {
            map[v1] = count++
        }

        if(!map[v2]) {
            map[v2] = count++
        }
    }

    const totalEqus = Array.from(new Array(count), () => [])
    for(let i=0; i<n; i++) {
        const [v1, v2] = equations[i];
        totalEqus[map[v1]].push([v2, values[i]]);
        totalEqus[map[v2]].push([v1, 1/values[i]]);
    }

    const result = []

    for(const [q1, q2] of queries) {
        let res = -1
        if(map[q1] != null && map[q2] != null) {
            if(q1 == q2) {
                res = 1
            } else {
                const queue = [q1]
                const divide = new Array(count).fill(-1);
                divide[map[q1]] = 1
                while(queue.length && divide[map[q2]] < 0) {
                    const div = queue.pop()
                    const allCase = totalEqus[map[div]]
                    for(const [divi, val] of allCase) {
                        if(divide[map[divi]] < 0) {
                            divide[map[divi]] = val * divide[map[div]]
                            queue.push(divi)
                        }
                    }
                }
                res = divide[map[q2]]
            }
        }
        result.push(res)
    }
    return result
};