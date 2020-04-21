
import javafx.util.Pair;

import java.util.*;


public class Main {

    public static void main(String[] args) {
        System.out.println(staNum(new int[]{1, 2, 3, 4, 5}, new int[]{3, 4, 5, 1, 2}));
    }


    /*
        82. Remove Duplicates from Sorted List II
        Time N
        Space 1
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode dummyHead = new ListNode(0);
        ListNode prev = dummyHead;
        ListNode cur = head;
        boolean flag = false;
        while (cur.next != null) {
            //当前位置已经重复
            if (flag) {
                if (cur.val != cur.next.val) {
                    flag = false;
                }
            }//当前位置没有重复
            else {
                if (cur.val == cur.next.val) {
                    flag = true;
                } else {
                    prev.next = cur;
                    prev = cur;
                }
            }
            cur = cur.next;
        }
        if (flag) {
            prev.next = null;
        } else {
            prev.next = cur;
        }
        return dummyHead.next;
    }

    /*
        byte dance
     */
    static int staNum(int[] gas, int[] cost) {
        int len = gas.length;
        int nowLeft = 0;
        int preCost = 0, flag = 0;//flag标记起点下标
        for (int i = 0; i < len; ++i) {
            nowLeft += gas[i] - cost[i];
            if (nowLeft < 0) {//此时油量为负数
                preCost += nowLeft;//累计负的油量
                nowLeft = 0;//清零重新开始
                flag = i + 1;//表示将下一个工区作为起点
            }
        }
        if (nowLeft + preCost < 0) {//总剩余油量为负数返回-1
            return -1;
        }
        return flag + 1;
    }

    /*
        525. Contiguous Array
        Time N
        Space N
     */
    public int findMaxLength(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int maxlen = 0, count = 0;
        for (int i = 0; i < nums.length; i++) {
            count = count + (nums[i] == 1 ? 1 : -1);
            if (map.containsKey(count)) {
                maxlen = Math.max(maxlen, i - map.get(count));
            } else {
                map.put(count, i);
            }
        }
        return maxlen;
    }

    /*
        174. Dungeon Game
        Time m*n
        space m*n
     */
    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length;
        if (m == 0) {
            return 0;
        }
        int n = dungeon[0].length;
        int[][] health = new int[m][n];
        int[][] dirs = new int[][]{{-1, 0}, {0, -1}};
        health[m - 1][n - 1] = 1 - dungeon[m - 1][n - 1] <= 0 ? 1 : 1 - dungeon[m - 1][n - 1];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                for (int[] dir : dirs) {
                    int newR = i + dir[0];
                    int newC = j + dir[1];
                    if (newR >= 0 && newC >= 0) {
                        int newHealth = health[i][j] - dungeon[newR][newC];
                        if (newHealth <= 0) {
                            newHealth = 1;
                        }
                        if (health[newR][newC] == 0) {
                            health[newR][newC] = newHealth;
                        } else {
                            health[newR][newC] = Math.min(newHealth, health[newR][newC]);
                        }
                    }
                }
            }
        }
        return health[0][0];
    }

    public int calculateMinimumHP1(int[][] dungeon) {
        int m = dungeon.length;
        if (m == 0) {
            return 0;
        }
        int n = dungeon[0].length;
        int[][] health = new int[m][n];
        Queue<int[]> bfs = new LinkedList<>();
        bfs.offer(new int[]{m - 1, n - 1});
        int[][] dirs = new int[][]{{-1, 0}, {0, -1}};
        health[m - 1][n - 1] = 1 - dungeon[m - 1][n - 1] <= 0 ? 1 : 1 - dungeon[m - 1][n - 1];
        Set<String> isVisited = new HashSet<>();
        while (!bfs.isEmpty()) {
            int[] cur = bfs.poll();
            StringBuilder sb = new StringBuilder();
            sb.append(cur[0]);
            sb.append("|");
            sb.append(cur[1]);
            isVisited.add(sb.toString());
            for (int[] dir : dirs) {
                int newR = cur[0] + dir[0];
                int newC = cur[1] + dir[1];
                if (newR >= 0 && newC >= 0) {
                    int newHealth = health[cur[0]][cur[1]] - dungeon[newR][newC];
                    if (newHealth <= 0) {
                        newHealth = 1;
                    }
                    if (health[newR][newC] == 0) {
                        health[newR][newC] = newHealth;
                    } else {
                        health[newR][newC] = Math.min(newHealth, health[newR][newC]);
                    }
                    StringBuilder sb1 = new StringBuilder();
                    sb1.append(newR);
                    sb1.append("|");
                    sb1.append(newC);
                    if (!isVisited.contains(sb1.toString())) {
                        bfs.offer(new int[]{newR, newC});
                    }
                }
            }
        }
        return health[0][0];
    }


    /*
        844. Backspace String Compare
        Time S + T
        Space S + T
     */
    public boolean backspaceCompare(String S, String T) {
        return build(S).equals(build(T));
    }

    public String build(String S) {
        Stack<Character> ans = new Stack();
        for (char c : S.toCharArray()) {
            if (c != '#') {
                ans.push(c);
            } else if (!ans.empty()) {
                ans.pop();
            }
        }
        return String.valueOf(ans);
    }


    /*
        84. Largest Rectangle in Histogram
        Time N
        Space N
     */
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> s = new Stack<>();
        int res = 0;
        for (int i = 0; i < heights.length; i++) {
            while (!s.isEmpty() && heights[s.peek()] > heights[i]) {
                int index = s.pop();
                if (!s.isEmpty()) {
                    res = Math.max(res, heights[index] * (i - s.peek() - 1));//
                } else {
                    res = Math.max(res, heights[index] * i);
                }
            }
            s.push(i);
        }
        while (!s.isEmpty()) {
            if (s.size() > 1) {
                int index = s.pop();
                res = Math.max(res, heights[index] * (heights.length - s.peek() - 1));
            } else {
                res = Math.max(res, heights[s.pop()] * heights.length);
            }
        }
        return res;
    }

    //遍历所有区间 找到区间里的最小值乘以区间长度
    //Time N^2
    public int largestRectangleAreaBF(int[] heights) {
        int maxarea = 0;
        for (int i = 0; i < heights.length; i++) {
            int minheight = Integer.MAX_VALUE;
            for (int j = i; j < heights.length; j++) {
                minheight = Math.min(minheight, heights[j]);
                maxarea = Math.max(maxarea, minheight * (j - i + 1));
            }
        }
        return maxarea;
    }

    /*
        136
        Single Number
        time N
        space 1
     */
    public int singleNumber(int[] nums) {
        int a = 0;
        for (int num : nums) {
            a ^= num;
        }
        return a;
    }

    /*
        148. Sort List
        Time NlogN
        space logn
     */
    public ListNode sortList(ListNode head) {
        if (head == null) {
            return null;
        }
        if (head.next != null) {
            ListNode mid = findMiddle(head);
            ListNode secHead = mid.next;
            mid.next = null;
            ListNode p1 = sortList(head);
            ListNode p2 = sortList(secHead);
            return mergeList(p1, p2);
        } else {
            return head;
        }

    }

    private ListNode mergeList(ListNode p1, ListNode p2) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (p1 != null && p2 != null) {
            if (p1.val < p2.val) {
                cur.next = p1;
                p1 = p1.next;
            } else {
                cur.next = p2;
                p2 = p2.next;
            }
            cur = cur.next;
        }
        if (p1 != null) {
            cur.next = p1;
        }
        if (p2 != null) {
            cur.next = p2;
        }
        return dummy.next;

    }

    private ListNode findMiddle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public static void merge(int[] a, int low, int mid, int high) {
        int[] temp = new int[high - low + 1];
        int i = low;// 左指针
        int j = mid + 1;// 右指针
        int k = 0;
        // 把较小的数先移到新数组中
        while (i <= mid && j <= high) {
            if (a[i] < a[j]) {
                temp[k++] = a[i++];
            } else {
                temp[k++] = a[j++];
            }
        }
        // 把左边剩余的数移入数组
        while (i <= mid) {
            temp[k++] = a[i++];
        }
        // 把右边边剩余的数移入数组
        while (j <= high) {
            temp[k++] = a[j++];
        }
        // 把新数组中的数覆盖nums数组
        for (int k2 = 0; k2 < temp.length; k2++) {
            a[k2 + low] = temp[k2];
        }
    }

    public static void mergeSort(int[] a, int low, int high) {
        int mid = (low + high) / 2;
        if (low < high) {
            // 左边
            mergeSort(a, low, mid);
            // 右边
            mergeSort(a, mid + 1, high);
            // 左右归并
            merge(a, low, mid, high);
            System.out.println(Arrays.toString(a));
        }

    }

    /*
        542. 01 Matrix
        Time N
        Space N
     */
    public int[][] updateMatrix(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return matrix == null ? null : matrix;
        }
        int row = matrix.length;
        int col = matrix[0].length;
        boolean[][] visited = new boolean[row][col];
        Queue<int[]> bfsHelper = new LinkedList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == 0) {
                    bfsHelper.offer(new int[]{i, j});
                }
            }
        }
        int dis = 1;
        int[][] dirs = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        while (!bfsHelper.isEmpty()) {
            int size = bfsHelper.size();
            for (int k = 0; k < size; k++) {
                int[] cur = bfsHelper.poll();
                for (int i = 0; i < 4; i++) {
                    int newRow = cur[0] + dirs[i][0];
                    int newCol = cur[1] + dirs[i][1];
                    if (inMatrix(newRow, newCol, row, col) &&
                        matrix[newRow][newCol] != 0 && !visited[newRow][newCol]) {
                        visited[newRow][newCol] = true;
                        matrix[newRow][newCol] = dis;
                        bfsHelper.offer(new int[]{newRow, newCol});
                    }
                }
            }
            dis++;
        }
        return matrix;
    }

    private boolean inMatrix(int r, int c, int row, int col) {
        return r >= 0 && r < row && c >= 0 && c < col;
    }

    /*
        1277. Count Square Submatrices with All Ones
        Time M*N
        Space 1
     */
    public int countSquares(int[][] A) {
        int res = 0;
        for (int i = 0; i < A.length; ++i) {
            for (int j = 0; j < A[0].length; ++j) {
                if (A[i][j] > 0 && i > 0 && j > 0) {
                    A[i][j] = Math.min(A[i - 1][j - 1], Math.min(A[i - 1][j], A[i][j - 1])) + 1;
                }
                res += A[i][j];
            }
        }
        return res;
    }

    /*
        1262. Greatest Sum Divisible by Three
        Time N
        Space 1
     */
    public int maxSumDivThree(int[] nums) {
        int[] remain = new int[]{0, Integer.MIN_VALUE, Integer.MIN_VALUE};
        for (int n : nums) {
            int[] newRemain = new int[3];
            for (int i = 0; i < 3; i++) {
                newRemain[(n + i) % 3] = Math
                    .max(remain[(n + i) % 3], n + remain[i]);
            }
            remain = newRemain;

        }
        return remain[0];
    }


    /*
        1268. Search Suggestions System
     */
    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        Arrays.sort(products);
        List<List<String>> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (char c : searchWord.toCharArray()) {
            sb.append(c);
            int k = Arrays.binarySearch(products, sb.toString());
            List<String> subRes = new ArrayList<>();
            for (int i = k; i < products.length && i < k + 3; i++) {
                if (products[i].indexOf(sb.toString()) != -1) {
                    subRes.add(products[i]);
                }
            }
            res.add(subRes);
        }
        return res;
    }


    /*
        1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
        Time M*N*min(M,N)
        space MN
     */
    public int maxSideLength(int[][] mat, int threshold) {
        int m = mat.length, n = mat[0].length;
        // Find prefix sum
        int[][] prefixSum = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            int sum = 0;
            for (int j = 1; j <= n; j++) {
                sum += mat[i - 1][j - 1];
                prefixSum[i][j] = prefixSum[i - 1][j] + sum;
            }
        }
        // Start from the largest side length
        for (int k = Math.min(m, n) - 1; k > 0; k--) {
            for (int i = 1; i + k <= m; i++) {
                for (int j = 1; j + k <= n; j++) {
                    if (prefixSum[i + k][j + k] - prefixSum[i - 1][j + k] - prefixSum[i + k][j - 1]
                        + prefixSum[i - 1][j - 1] <= threshold) {
                        return k + 1;
                    }
                }
            }
        }
        return 0;
    }

    /*
       1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
       边累加边判断
       Time M*N
       space MN
    */
    public int maxSideLengthOpt(int[][] mat, int threshold) {
        int m = mat.length;
        int n = mat[0].length;
        int[][] sum = new int[m + 1][n + 1];

        int res = 0;
        int len = 1; // square side length

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + mat[i - 1][j - 1];

                if (i >= len && j >= len
                    && sum[i][j] - sum[i - len][j] - sum[i][j - len] + sum[i - len][j - len]
                    <= threshold) {
                    res = len++;
                }
            }
        }

        return res;
    }


    /*
        1349. Maximum Students Taking Exam
        2^(MN)
        2^(MN)
     */
    int m, n;
    Map<String, Integer> memo;

    public int maxStudents(char[][] seats) {
        m = seats.length;
        if (m == 0) {
            return 0;
        }
        n = seats[0].length;

        memo = new HashMap<>();
        StringBuilder sb = new StringBuilder();
        for (char[] row : seats) {
            sb.append(row);
        }

        return dfs(sb.toString());
    }

    /* dfs returns the max student we can place if start with the given state */

    public int dfs(String state) {
        if (memo.containsKey(state)) {
            return memo.get(state);
        }
        int max = 0;
        char[] C = state.toCharArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                //we see an empty seat, there are two choices, place a student here or leave it empty.
                if (C[i * n + j] == '.') {
                    //choice (1): we choose not to place a student, but we place a x to mark this seat as unanvailable
                    // so we don't repeatedly search this state again.

                    C[i * n + j] = 'x';
                    max = Math.max(max, dfs(new String(C)));

                    //choice (2): we place a student, but this makes left, right, bottom left, bottom right seat unavailable.
                    if (j + 1 < n) {
                        if (i < m - 1 && C[(i + 1) * n + j + 1] == '.') {
                            C[(i + 1) * n + j + 1] = 'x';
                        }
                        if (C[i * n + j + 1] == '.') {
                            C[i * n + j + 1] = 'x';
                        }
                    }
                    if (j - 1 >= 0) {
                        if (i < m - 1 && C[(i + 1) * n + j - 1] == '.') {
                            C[(i + 1) * n + j - 1] = 'x';
                        }
                        if (C[i * n + j - 1] == '.') {
                            C[i * n + j - 1] = 'x';
                        }
                    }
                    max = Math.max(max, 1 + dfs(new String(C)));
                }
            }
        }
        memo.put(state, max);
        return max;
    }


    /*
        1358. Number of Substrings Containing All Three Characters
        Time N
        O 1
     */
    public int numberOfSubstrings(String s) {
        int last[] = {-1, -1, -1}, count = 0, n = s.length();
        for (int i = 0; i < n; i++) {
            last[s.charAt(i) - 'a'] = i;
            int startIndex = Math.min(last[0], Math.min(last[1], last[2]));
            if (startIndex != -1) {
                count += 1 + startIndex;
            }
        }
        return count;
    }

    /*
        1353. Maximum Number of Events That Can Be Attended
        Time NlogN
        Space N
     */
    public int maxEvents(int[][] events) {
        if (events == null || events.length == 0) {
            return 0;
        }
        Arrays.sort(events, (a, b) -> (a[0] - b[0]));
        PriorityQueue<Integer> minheap = new PriorityQueue<>();
        int count = 0;
        int curDate = 0;
        int curEvent = 0;
        int maxDate = events[events.length - 1][1];
        while (!minheap.isEmpty() || curEvent < events.length) {
            if (minheap.isEmpty()) {
                curDate = events[curEvent][0];
            }
            while (curEvent < events.length && events[curEvent][0] == curDate) {
                minheap.add(events[curEvent][1]);
                curEvent++;
            }
            minheap.poll();
            count++;
            curDate++;
            while (!minheap.isEmpty() && minheap.peek() < curDate) {
                minheap.poll();
            }
        }

        return count;
    }

    /*
        991. Broken Calculator
        We do Y/2 all the way until it's smaller than X,
        time complexity is O(log(Y/X)).
     */
    public int brokenCalc(int X, int Y) {
        int steps = 0;
        while (Y > X) {
            steps++;
            if ((Y & 1) == 0) {
                Y = Y >> 1;
            } else {
                Y++;
            }
        }
        return steps + X - Y;
    }

    /*
        53. Maximum Subarray
        Time N
        Space 1
     */
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int currSum = nums[0], maxSum = nums[0];

        for (int i = 1; i < n; ++i) {
            currSum = Math.max(nums[i], currSum + nums[i]);
            maxSum = Math.max(maxSum, currSum);
        }
        return maxSum;
    }



    /*
        286. Walls and Gates
        Time MN
        Space MN
     */

    public void wallsAndGates(int[][] rooms) {
        int row = rooms.length;
        if (row == 0) {
            return;
        }

        List<int[]> roomList = new ArrayList<>();
        int col = rooms[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (rooms[i][j] == 0) {
                    roomList.add(new int[]{i, j});
                }
            }
        }
        bfsFromRoom(rooms, roomList);
    }

    int[][] dir = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

    private void bfsFromRoom(int[][] rooms, List<int[]> roomList) {
        Queue<int[]> bfsHelper = new LinkedList<>();
        for (int[] room : roomList) {
            bfsHelper.add(room);
        }
        int level = 0;
        int col = rooms[0].length;
        while (!bfsHelper.isEmpty()) {
            int size = bfsHelper.size();
            for (int i = 0; i < size; i++) {
                int[] cur = bfsHelper.poll();
                if (rooms[cur[0]][cur[1]] != 0 && rooms[cur[0]][cur[1]] != Integer.MAX_VALUE) {
                    continue;
                }
                rooms[cur[0]][cur[1]] = level;
                for (int d = 0; d < 4; d++) {
                    int[] direction = dir[d];
                    int newRow = cur[0] + direction[0];
                    int newCol = cur[1] + direction[1];
                    if (inGrid(newRow, newCol, rooms)
                        && rooms[newRow][newCol] == Integer.MAX_VALUE) {
                        bfsHelper.add(new int[]{newRow, newCol});
                    }
                }
            }
            level++;
        }
    }

    /*
        387. First Unique Character in a String
        Time N
        Space N
     */
    public int firstUniqChar(String s) {
        Map<Character, Integer> dict = new HashMap<>();
        for (char c : s.toCharArray()) {
            dict.put(c, dict.getOrDefault(c, 0) + 1);
        }
        for (int i = 0; i < s.length(); i++) {
            if (dict.get(s.charAt(i)) == 1) {
                return i;
            }
        }
        return -1;
    }

    /*
        242. Valid Anagram
     */
    public boolean isAnagram(String s, String t) {
        Map<Character, Integer> dict = new HashMap<>();
        for (char c : s.toCharArray()) {
            dict.put(c, dict.getOrDefault(c, 0) + 1);
        }

        for (char c : t.toCharArray()) {
            if (!dict.containsKey(c)) {
                return false;
            }
            int freq = dict.get(c);
            if (freq == 1) {
                dict.remove(c);
            } else {
                dict.put(c, freq - 1);
            }
        }
        return dict.isEmpty();
    }

    /*
        38. Count and Say
        Time 2^n
        space 2^n;
     */
    public String countAndSay(int n) {
        String prev = "1";
        if (n == 1) {
            return prev;
        }
        StringBuilder next = null;
        for (int i = 1; i < n; i++) {
            next = new StringBuilder();
            for (int j = 0; j < prev.length(); ) {
                int count = 1;
                char cur = prev.charAt(j);
                while (++j < prev.length() && cur == prev.charAt(j)) {
                    count++;
                }
                next.append(count);
                next.append(cur);
            }
            prev = next.toString();
        }
        return next.toString();
    }

    /*
        65. Valid Number
        N
        1
     */
    public boolean isNumber(String s) {
        if (s == null) {
            return false;
        }
        s = s.trim();
        boolean hasDot = false;
        boolean isBegin = false;
        boolean hasE = false;
        boolean hasNumberBeforeE = false;
        boolean hasNumberAfterDot = false;
        boolean hasNumberBeforeDot = false;
        boolean hasNumberAfterE = false;
        boolean hasSignAfterE = false;
        for (int i = 0; i < s.length(); i++) {
            char cur = s.charAt(i);
            if (cur == '+' || cur == '-') {
                if (isBegin && s.charAt(i - 1) == 'e' && !hasSignAfterE) {
                    hasSignAfterE = true;
                } else if (!isBegin) {
                    isBegin = true;
                } else {
                    return false;
                }
            } else if (Character.isDigit(cur)) {
                if (!hasE) {
                    hasNumberBeforeE = true;
                }
                if (hasE) {
                    hasNumberAfterE = true;
                }
                if (!hasDot) {
                    hasNumberBeforeDot = true;
                } else if (s.charAt(i - 1) == '.') {
                    hasNumberAfterDot = true;
                }
            } else if (cur == '.') {
                if (hasDot || hasE) {
                    return false;
                }
                hasDot = true;
            } else if (cur == 'e') {
                if (!hasNumberBeforeE || hasE) {
                    return false;
                } else {
                    hasE = true;
                }
            } else {
                return false;
            }
            isBegin = true;
        }
        if (!isBegin) {
            return false;
        }
        if (hasE) {
            if (!hasNumberAfterE) {
                return false;
            }
        }

        if (hasDot) {
            if (!hasNumberBeforeDot && !hasNumberAfterDot) {
                return false;
            }
        }

        return true;


    }

    /*
        68. Text Justification
        Time O(N)
        Space O(N)
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        int cur = 0;
        //go through words list
        while (cur < words.length) {
            //start a new line
            int curLength = 0;
            List<String> curList = new ArrayList<>();
            //while this new line is not full
            while (cur < words.length && curLength < maxWidth) {
                //if this is a brand new line just add the word
                //or it will add the word and a space in the middle
                if ((curLength == 0 && words[cur].length() <= maxWidth)
                    || curLength + words[cur].length() + 1 <= maxWidth) {
                    curList.add(words[cur]);
                    curLength += curLength == 0 ? words[cur].length() : words[cur].length() + 1;
                    cur++;
                }
                // if adding next word would over this line's size,
                // end adding words to the current line.
                else {
                    break;
                }
            }
            //calculate the space in the middle and space in the right side.
            //let the curLength represents the total length of letters.
            curLength -= curList.size() - 1;
            int space = maxWidth - curLength;
            //space between each two words
            int spaceBetweenLength;
            //rest space should be assign from left
            int restSpace;
            //if there is only 1 words, assign the space specifically.
            if (curList.size() - 1 == 0) {
                spaceBetweenLength = space;
                restSpace = 0;
            } else {
                spaceBetweenLength = space / (curList.size() - 1);
                restSpace = space % (curList.size() - 1);
            }
            //Build the space String
            StringBuilder spaceBetween = new StringBuilder();
            for (int i = 0; i < spaceBetweenLength; i++) {
                spaceBetween.append(' ');
            }
            String middleSpace = spaceBetween.toString();
            StringBuilder line = new StringBuilder();
            //if this line is the last line. it should be left-justified.
            if (cur == words.length) {
                StringBuilder followSpace = new StringBuilder();
                for (int i = 0; i < maxWidth - curLength - curList.size() + 1; i++) {
                    followSpace.append(' ');
                }
                for (int i = 0; i < curList.size() - 1; i++) {
                    line.append(curList.get(i));
                    line.append(' ');
                }
                line.append(curList.get(curList.size() - 1));
                line.append(followSpace);
            }
            //At normal line, padding the space evenly.
            else {
                for (int i = 0; i < curList.size(); i++) {
                    line.append(curList.get(i));
                    if (i < curList.size() - 1 || curList.size() == 1) {
                        line.append(middleSpace);
                    }
                    if (i < restSpace) {
                        line.append(' ');
                    }
                }
            }

            res.add(line.toString());
        }
        return res;
    }


    /*
        150. Evaluate Reverse Polish Notation
        Time N
        Space N
     */
    public int evalRPN(String[] tokens) {
        Stack<String> helper = new Stack<>();
        for (String token : tokens) {
            if (isOperator(token)) {
                int res = 0;
                try {
                    int b = Integer.parseInt(helper.pop());
                    int a = Integer.parseInt(helper.pop());
                    switch (token.charAt(0)) {
                        case '+':
                            res = a + b;
                            break;
                        case '-':
                            res = a - b;
                            break;
                        case '*':
                            res = a * b;
                            break;
                        case '/':
                            if (b == 0) {
                                throw new RuntimeException("The divisor can't be 0");
                            }
                            res = a / b;
                            break;
                    }
                } catch (EmptyStackException e) {
                    throw new RuntimeException("Invalid Expression");
                }
                helper.push(String.valueOf(res));
            } else {
                helper.push(token);
            }
        }
        return Integer.parseInt(helper.peek());
    }

    private boolean isOperator(String c) {
        return "+".equals(c) || "-".equals(c) || "*".equals(c) || "/".equals(c);
    }

    /*
        406. Queue Reconstruction by Height
        Time o N^3 下面有N^2的解法
        Space N

     */
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (o1, o2) -> o1[1] == o2[1] ? o1[0] - o2[0] : o1[1] - o2[1]);
        List<int[]> peopleList = new ArrayList<>();

        outer:
        for (int i = 0; i < people.length; i++) {
            int[] curPerson = people[i];
            int count = 0;
            for (int j = 0; j < peopleList.size(); j++) {
                //meet the k taller person in front. place this person.
                if (count == curPerson[1]) {
                    while (j < peopleList.size() && peopleList.get(j)[0] < curPerson[0]
                        && peopleList.get(j)[1] == curPerson[1]) {
                        j++;
                    }
                    //O(N) operation
                    peopleList.add(j, curPerson);
                    continue outer;
                }

                if (peopleList.get(j)[0] >= curPerson[0]) {
                    count++;
                }
            }
            //add to the end of the queue.
            if (count == curPerson[1]) {
                peopleList.add(curPerson);
            }
            //no solution
            else if (count < curPerson[1]) {
                return null;
            }
        }
        return peopleList.toArray(new int[people.length][2]);
    }

    //从高到矮排队，然后每次根据前面的人数来直接插入到list
    public int[][] reconstructQueueOPT(int[][] people) {

        // if the heights are equal, compare k-values
        Arrays.sort(people, (o1, o2) -> o1[0] == o2[0] ? o1[1] - o2[1] : o2[0] - o1[0]);

        List<int[]> output = new ArrayList<>();
        for (int[] p : people) {
            output.add(p[1], p);
        }

        int n = people.length;
        return output.toArray(new int[n][2]);
    }

    /*
        234. Palindrome Linked List
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return false;
        }
        if (head.next == null) {
            return true;
        }
        ListNode slow = head;
        ListNode fast = head;
        ListNode prev = null;
        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        //list size is even number
        //slow pointer is at the second middle point
        ListNode secondHalfHead = null;
        prev.next = null;
        if (fast == null) {
            secondHalfHead = reverseListIterative(slow);
        }
        //list size is odd number
        //slow pointer is right at the middle
        else {
            secondHalfHead = reverseListIterative(slow.next);
        }
        while (secondHalfHead != null) {
            if (secondHalfHead.val != head.val) {
                return false;
            }
            secondHalfHead = secondHalfHead.next;
            head = head.next;
        }
        return true;
    }

    /*
        86. Partition List
        Time N
        Space 1
     */
    public ListNode partition(ListNode head, int x) {
        ListNode dummy = new ListNode(0);
        ListNode cur = head;
        ListNode subTail = null;
        //pass all the value less than x
        while (cur != null) {
            if (cur.val < x) {
                if (dummy.next == null) {
                    dummy.next = cur;
                }
                subTail = cur;
                cur = cur.next;
            } else {
                break;
            }
        }
        ListNode prev = subTail;
        //find a value
        while (cur != null) {
            if (cur.val < x) {
                ListNode node = cur;
                //insert it to head
                if (dummy.next == null) {
                    prev.next = node.next;
                    dummy.next = node;
                    node.next = head;
                }
                //insert it after tail
                else {
                    prev.next = node.next;
                    ListNode subHead = subTail.next;
                    subTail.next = node;
                    node.next = subHead;
                }
            }
            prev = cur;
            cur = cur.next;
        }

        return dummy.next;

    }

    /*
        92. Reverse Linked List II
        Time O(1)
        Space O(1)

     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null || m == n) {
            return head;
        }
        if (m > 1) {
            ListNode cur = head;
            ListNode prev = null;
            for (int i = 0; i < m - 1; i++) {
                prev = cur;
                cur = cur.next;
            }
            ListNode frontTail = prev;
            ListNode subTail = cur;
            prev = cur;
            cur = cur.next;
            for (int i = 0; i < n - m; i++) {
                ListNode tempNext = cur.next;
                cur.next = prev;
                prev = cur;
                cur = tempNext;
                if (cur == null) {
                    break;
                }
            }
            subTail.next = cur;
            frontTail.next = prev;
            return head;
        } else {
            ListNode prev = null;
            ListNode cur = head;
            for (int i = 0; i < n; i++) {
                ListNode tempNext = cur.next;
                cur.next = prev;
                prev = cur;
                cur = tempNext;
                if (cur == null) {
                    break;
                }
            }
            if (cur != null) {
                head.next = cur;
            }
            return prev;
        }

    }

    /*
         206. Reverse Linked List
         Time N
         Space 1
     */
    public ListNode reverseListIterative(ListNode head) {

        if (head == null) {
            return null;
        }
        ListNode prev = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode tempNext = cur.next;
            cur.next = prev;
            prev = cur;
            cur = tempNext;
        }

        return prev;
    }

    /*
        206. Reverse Linked List
         Time N
         Space N
     */

    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode n = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return n;
    }

    /*
        876. Middle of the Linked List
        Time O(N)
        Space O(1)
     */
    public ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    /*
        912. Sort an Array
        time nlogn
        space logn
     */
    class qSort {

        private int[] nums;

        public List<Integer> sortArray(int[] nums) {
            this.nums = nums;
            qSort(0, nums.length - 1);
            List<Integer> res = new ArrayList<>();
            for (int num : nums) {
                res.add(num);
            }
            return res;
        }

        private void qSort(int start, int end) {
            if (start < end) {
                int pivotIndex = divide(start, end);
                qSort(start, pivotIndex);
                qSort(pivotIndex + 1, end);
            }
        }

        private int divide(int i, int j) {
            int pivot = nums[i];

            while (i < j) {
                while (j > i && nums[j] >= pivot) {
                    j--;
                }
                if (j > i) {
                    nums[i] = nums[j];
                    i++;
                }
                while (i < j && nums[i] < pivot) {
                    i++;
                }
                if (i < j) {
                    nums[j] = nums[i];
                    j--;
                }
            }
            nums[i] = pivot;
            return i;
        }
    }


    /*
        1296. Divide Array in Sets of K Consecutive Numbers
        Time  max(dlogd,n)  d is the number of distinct elements(size of the heap)
        Space o(N)
     */
    public boolean isPossibleDivide(int[] nums, int k) {
        int len = nums.length;
        if (len % k != 0) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<>();
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int n : nums) {
            map.put(n, map.getOrDefault(n, 0) + 1);
        }
        for (int n : map.keySet()) {
            pq.add(n);
        }
        while (!pq.isEmpty()) {
            int cur = pq.poll();
            if (map.get(cur) == 0) {
                continue;
            }
            int times = map.get(cur);
            for (int i = 0; i < k; i++) {
                if (!map.containsKey(cur + i) || map.get(cur + i) < times) {
                    return false;
                }
                map.put(cur + i, map.get(cur + i) - times);
            }
            len -= k * times;
        }
        return len == 0;
    }

    /*
           628. Maximum Product of Three Numbers
           Time O(N)
           Space O(1)
           找到三个最大的正数 或者两个最大的负数
           比较两负一正 和三正哪个大
    */
    public int maximumProduct(int[] nums) {
        int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE;
        for (int n : nums) {
            if (n <= min1) {
                min2 = min1;
                min1 = n;
            } else if (n <= min2) {     // n lies between min1 and min2
                min2 = n;
            }
            if (n >= max1) {            // n is greater than max1, max2 and max3
                max3 = max2;
                max2 = max1;
                max1 = n;
            } else if (n >= max2) {     // n lies betweeen max1 and max2
                max3 = max2;
                max2 = n;
            } else if (n >= max3) {     // n lies betwen max2 and max3
                max3 = n;
            }
        }
        return Math.max(min1 * min2 * max1, max1 * max2 * max3);
    }


    public static boolean arraySumInAnotherArray(int[] a, int[] b) {
        Set<Integer> setB = new HashSet<>();
        for (int num : b) {
            setB.add(num);
        }
        for (int i = 0; i < a.length - 1; i++) {
            for (int j = i + 1; j < a.length; j++) {
                if (setB.contains(a[i] + a[j])) {
                    return true;
                }
            }
        }
        return false;
    }


    /*
        152. Maximum Product Subarray
        每次储存乘到上一个数的最小和最大值，更新当前数乘上个数或者不乘的当前最小最大值
        更新最大答案
        Time O(N)
        Space O(1)
     */
    public int maxProduct(int[] nums) {
        int prevMax = nums[0], prevMin = nums[0];
        int res = prevMax;
        for (int i = 1; i < nums.length; i++) {
            int tempMax = Math.max(nums[i], Math.max(nums[i] * prevMax, nums[i] * prevMin));
            prevMin = Math.min(nums[i], Math.min(nums[i] * prevMax, nums[i] * prevMin));
            prevMax = tempMax;
            res = Math.max(res, prevMax);
        }
        return res;
    }

    public static int findDateInAYear(int year) {
        Calendar c = new GregorianCalendar();
        c.set(year, Calendar.OCTOBER, 1);
        int diff = c.get(Calendar.DAY_OF_WEEK) - Calendar.TUESDAY;
        if (diff > 0) {
            int offSet = 14 - diff;
            c.set(year, Calendar.OCTOBER, offSet + 1);
        } else {
            c.set(year, Calendar.OCTOBER, 8 - diff);
        }
        System.out.println(c.getTime());
        return c.get(Calendar.DAY_OF_MONTH);
    }

    /*
        164. Maximum Gap
        Time O(N+ gap)
        Space O(gap)
        首先将min 和max 两个数字拿出来
        将N-2个数字 平均分到N-1个桶中
        其中必有一个桶是空的
        所以我们的最大gap已经是在桶与桶之间，而不在桶的内部
        只需要一次顺序扫描桶 找到最大gap
     */
    public int maximumGap(int[] num) {
        if (num == null || num.length < 2) {
            return 0;
        }
        // get the max and min value of the array
        int min = num[0];
        int max = num[0];
        for (int i : num) {
            min = Math.min(min, i);
            max = Math.max(max, i);
        }
        // the minimum possibale gap, ceiling of the integer division
        int gap = (int) Math.ceil((double) (max - min) / (num.length - 1));
        int[] bucketsMIN = new int[num.length - 1]; // store the min value in that bucket
        int[] bucketsMAX = new int[num.length - 1]; // store the max value in that bucket
        Arrays.fill(bucketsMIN, Integer.MAX_VALUE);
        Arrays.fill(bucketsMAX, Integer.MIN_VALUE);
        // put numbers into buckets
        for (int i : num) {
            if (i == min || i == max) {
                continue;
            }
            int idx = (i - min) / gap; // index of the right position in the buckets
            bucketsMIN[idx] = Math.min(i, bucketsMIN[idx]);
            bucketsMAX[idx] = Math.max(i, bucketsMAX[idx]);
        }
        // scan the buckets for the max gap
        int maxGap = Integer.MIN_VALUE;
        int previous = min;
        for (int i = 0; i < num.length - 1; i++) {
            if (bucketsMIN[i] == Integer.MAX_VALUE && bucketsMAX[i] == Integer.MIN_VALUE)
            // empty bucket
            {
                continue;
            }
            // min value minus the previous value is the current gap
            maxGap = Math.max(maxGap, bucketsMIN[i] - previous);
            // update previous bucket value
            previous = bucketsMAX[i];
        }
        maxGap = Math.max(maxGap, max - previous); // updata the final max value gap
        return maxGap;
    }

    /*
        735. Asteroid Collision
        Time O(N)
        Space O(N)
     */
    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> stack = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        for (int asteroid : asteroids) {
            if (asteroid > 0) {
                stack.addLast(asteroid);
            } else {
                if (stack.isEmpty()) {
                    res.add(asteroid);
                } else {
                    while (!stack.isEmpty() && -asteroid > stack.peek()) {
                        stack.removeLast();
                    }
                    if (stack.isEmpty()) {
                        res.add(asteroid);
                    } else if (stack.peek() == -asteroid) {
                        stack.removeLast();
                    }
                }
            }
        }
        res.addAll(stack);
        int[] finalResult = new int[res.size()];
        for (int i = 0; i < finalResult.length; i++) {
            finalResult[i] = res.get(i);
        }
        return finalResult;
    }

    /*
        414. Third Maximum Number
     */

    public int thirdMax(int[] nums) {
        Integer max1 = null;
        Integer max2 = null;
        Integer max3 = null;
        for (Integer n : nums) {
            if (n.equals(max1) || n.equals(max2) || n.equals(max3)) {
                continue;
            }
            if (max1 == null || n > max1) {
                max3 = max2;
                max2 = max1;
                max1 = n;
            } else if (max2 == null || n > max2) {
                max3 = max2;
                max2 = n;
            } else if (max3 == null || n > max3) {
                max3 = n;
            }
        }
        return max3 == null ? max1 : max3;

    }

    /*
        274. H-Index
        bucket sort
        Time O(N)
        space O(N)
    */
    public int hIndex(int[] citations) {
        int n = citations.length;
        int[] papers = new int[n + 1];
        // counting papers for each citation number
        for (int c : citations) {
            papers[Math.min(n, c)]++;
        }
        // finding the h-index
        int s = 0;
        for (int k = n; k >= 0; k--) {
            s += papers[k];
            if (s >= k) {
                return k;
            }
        }
        return -1;
    }


    public int minSteps(String s, String t) {
        if (s.length() == 0) {
            return 0;
        }
        Map<Character, Integer> count = new HashMap<>();
        for (char c : s.toCharArray()) {
            count.put(c, count.getOrDefault(c, 0) + 1);
        }
        int minSteps = 0;
        for (char c : t.toCharArray()) {
            if (!count.containsKey(c)) {
                minSteps++;
            } else {
                int curCount = count.get(c);
                if (curCount == 1) {
                    count.remove(c);
                } else {
                    count.put(c, curCount - 1);
                }
            }
        }
        return minSteps;
    }

    public boolean checkIfExist(int[] arr) {
        if (arr.length < 2) {
            return false;
        }
        Set<Integer> s = new HashSet<>();
        for (int num : arr) {
            if (num == 0 && s.contains(0)) {
                return true;
            }
            s.add(num);
        }
        for (int num : arr) {
            s.remove(num);
            if (s.contains(num * 2)) {
                return true;
            }
            s.add(num);
        }
        return false;
    }

    /*
        994. Rotting Oranges
        N is the number of cells
        time o(N)
        space O(N)
     */
    public int orangesRotting(int[][] grid) {
        int row = grid.length;
        if (row == 0) {
            return 0;
        }
        int col = grid[0].length;
        List<int[]> rottingOranges = new LinkedList<>();
        int freshCount = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    freshCount++;
                }
                if (grid[i][j] == 2) {
                    rottingOranges.add(new int[]{i, j});
                }
            }
        }
        int steps = 0;
        int[][] offset = new int[][]{{0, 1}, {1, 0}, {-1, 0}, {0, -1}};
        outer:
        while (!rottingOranges.isEmpty() && freshCount != 0) {
            List<int[]> nextTurn = new LinkedList<>();
            steps++;
            for (int[] orange : rottingOranges) {
                for (int i = 0; i < 4; i++) {
                    int newRow = orange[0] + offset[i][0];
                    int newCol = orange[1] + offset[i][1];
                    if (inGrid(newRow, newCol, grid) && grid[newRow][newCol] == 1) {
                        grid[newRow][newCol] = 2;
                        nextTurn.add(new int[]{newRow, newCol});
                        freshCount--;
                        if (freshCount == 0) {
                            break outer;
                        }
                    }
                }
            }
            rottingOranges = nextTurn;
        }

        return freshCount == 0 ? steps : -1;
    }


    /*
        198. House Robber
        dp
        time O(N)
        space O(1)
     */
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int prev = nums[0];
        int cur = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            int temp = Math.max(prev + nums[i], cur);
            prev = cur;
            cur = temp;
        }
        return cur;
    }

    /*
        738. Monotone Increasing Digits
        Time O(D) ≈ log(N)
        Space Complexity: O(D), the size of the answer.
     */
    public int monotoneIncreasingDigits(int N) {
        String num = String.valueOf(N);
        int digits = num.length();
        int i = 0, j = 0;
        char[] res = num.toCharArray();
        while (j < digits - 1) {
            if (res[j] > res[j + 1]) {
                break;
            } else if (res[j] == res[j + 1]) {
                j++;
            } else {
                j++;
                i = j;
            }
        }
        if (i == digits - 1) {
            return N;
        } else {
            res[i]--;
            i++;
            for (; i < digits; i++) {
                res[i] = '9';
            }
            return Integer.parseInt(String.valueOf(res));
        }
    }

    /*
        189. Rotate Array
        Original List                   : 1 2 3 4 5 6 7
        After reversing all numbers     : 7 6 5 4 3 2 1
        After reversing first k numbers : 5 6 7 4 3 2 1
        After revering last n-k numbers : 5 6 7 1 2 3 4 --> Result
        Time O(n)
        Space O(1)
     */
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }


    /*
        140. Word Break II
        Time O(N^3)
        space O(n^3)
     */
    public List<String> wordBreak2(String s, List<String> wordDict) {
        Map<Integer, List<String>> map = new HashMap<>();
        return word_Break(s, new HashSet<>(wordDict), 0, map);
    }


    public List<String> word_Break(String s, Set<String> wordDict, int start,
        Map<Integer, List<String>> map) {
        if (map.containsKey(start)) {
            return map.get(start);
        }
        LinkedList<String> res = new LinkedList<>();
        if (start == s.length()) {
            res.add("");
        }
        for (int end = start + 1; end <= s.length(); end++) {
            if (wordDict.contains(s.substring(start, end))) {
                List<String> list = word_Break(s, wordDict, end, map);
                for (String l : list) {
                    res.add(s.substring(start, end) + (l.equals("") ? "" : " ") + l);
                }
            }
        }
        map.put(start, res);
        return res;
    }

    /*
        139. Word Break
        Time O(n^2)
        space O(N)
         The depth of recursion tree can go up to n.
         下面还有一个dp 解 同样时空复杂度

     */

    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> dict = new HashSet<>();
        dict.addAll(wordDict);
        return checkSubString(s, 0, dict, new int[s.length()]);
    }

    private boolean checkSubString(String s, int index, Set<String> dict, int[] memo) {
        if (index == s.length()) {
            return true;
        }
        //memo array is used to store the result of a specific substring.
        //avoid repeated computation
        if (memo[index] != 0) {
            return memo[index] == 1;
        }
        int cur = index + 1;
        while (cur <= s.length()) {
            if (dict.contains(s.substring(index, cur))) {
                if (checkSubString(s, cur, dict, memo)) {
                    memo[index] = 1;
                    return true;
                }
            }
            cur++;
        }
        memo[index] = -1;
        return false;
    }


    public boolean wordBreakDP(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    public String one(int num) {
        switch (num) {
            case 1:
                return "One";
            case 2:
                return "Two";
            case 3:
                return "Three";
            case 4:
                return "Four";
            case 5:
                return "Five";
            case 6:
                return "Six";
            case 7:
                return "Seven";
            case 8:
                return "Eight";
            case 9:
                return "Nine";
        }
        return "";
    }

    public String twoLessThan20(int num) {
        switch (num) {
            case 10:
                return "Ten";
            case 11:
                return "Eleven";
            case 12:
                return "Twelve";
            case 13:
                return "Thirteen";
            case 14:
                return "Fourteen";
            case 15:
                return "Fifteen";
            case 16:
                return "Sixteen";
            case 17:
                return "Seventeen";
            case 18:
                return "Eighteen";
            case 19:
                return "Nineteen";
        }
        return "";
    }

    public String ten(int num) {
        switch (num) {
            case 2:
                return "Twenty";
            case 3:
                return "Thirty";
            case 4:
                return "Forty";
            case 5:
                return "Fifty";
            case 6:
                return "Sixty";
            case 7:
                return "Seventy";
            case 8:
                return "Eighty";
            case 9:
                return "Ninety";
        }
        return "";
    }

    public String two(int num) {
        if (num == 0) {
            return "";
        } else if (num < 10) {
            return one(num);
        } else if (num < 20) {
            return twoLessThan20(num);
        } else {
            int tenner = num / 10;
            int rest = num - tenner * 10;
            if (rest != 0) {
                return ten(tenner) + " " + one(rest);
            } else {
                return ten(tenner);
            }
        }
    }

    public String three(int num) {
        int hundred = num / 100;
        int rest = num - hundred * 100;
        String res = "";
        if (hundred * rest != 0) {
            res = one(hundred) + " Hundred " + two(rest);
        } else if ((hundred == 0) && (rest != 0)) {
            res = two(rest);
        } else if ((hundred != 0) && (rest == 0)) {
            res = one(hundred) + " Hundred";
        }
        return res;
    }

    /*
        273. Integer to English Words
        Time complexity : O(N). Intuitively the output is proportional
        to the number N of digits in the input.
        Space complexity : O(1) since the output is just a string.
     */
    public String numberToWords(int num) {
        if (num == 0) {
            return "Zero";
        }

        int billion = num / 1000000000;
        int million = (num - billion * 1000000000) / 1000000;
        int thousand = (num - billion * 1000000000 - million * 1000000) / 1000;
        int rest = num - billion * 1000000000 - million * 1000000 - thousand * 1000;

        String result = "";
        if (billion != 0) {
            result = three(billion) + " Billion";
        }
        if (million != 0) {
            if (!result.isEmpty()) {
                result += " ";
            }
            result += three(million) + " Million";
        }
        if (thousand != 0) {
            if (!result.isEmpty()) {
                result += " ";
            }
            result += three(thousand) + " Thousand";
        }
        if (rest != 0) {
            if (!result.isEmpty()) {
                result += " ";
            }
            result += three(rest);
        }
        return result;
    }

    /*
        269. Alien Dictionary
        Input:
        [
          "wrt",
          "wrf",
          "er",
          "ett",
          "rftt"
        ]

        Output: "wertf"

        Time : O(N+M) N for number of all letters M for number of words
        space O(N+M)
     */
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> edge = new HashMap<>();
        StringBuilder res = new StringBuilder();
        for (String word : words) {
            for (char c : word.toCharArray()) {
                edge.put(c, new HashSet<>());
            }
        }
        for (int wordIndex = 0; wordIndex < words.length - 1; wordIndex++) {
            String wordA = words[wordIndex];
            String wordB = words[wordIndex + 1];
            int cur = 0;
            while (cur < wordA.length() && cur < wordB.length()) {
                if (wordA.charAt(cur) == wordB.charAt(cur)) {
                    cur++;
                    continue;
                }
                Set<Character> prevSet = edge.get(wordB.charAt(cur));
                prevSet.add(wordA.charAt(cur));
                edge.put(wordB.charAt(cur), prevSet);
                break;
            }

        }
        Queue<Character> readyForSort = new LinkedList<Character>();
        for (Map.Entry<Character, Set<Character>> e : edge.entrySet()) {
            if (e.getValue().isEmpty()) {
                readyForSort.add(e.getKey());
            }
        }
        int resLength = edge.size();

        while (!readyForSort.isEmpty()) {
            char curChar = readyForSort.poll();
            res.append(curChar);
            for (Map.Entry<Character, Set<Character>> e : edge.entrySet()) {
                if (e.getValue().contains(curChar)) {
                    e.getValue().remove(curChar);
                    if (e.getValue().isEmpty()) {
                        readyForSort.add(e.getKey());
                    }
                }
            }
        }
        return res.length() == resLength ? res.toString() : "";
    }

    /*
        304. Range Sum Query 2D - Immutable
     */
    static class NumMatrix1 {

        private int[][] sumMatrix;

        //time O(N*M)
        public NumMatrix1(int[][] matrix) {
            int row = matrix.length;
            int col = row == 0 ? 0 : matrix[0].length;
            sumMatrix = new int[row][col];
            if (row != 0) {
                for (int i = 0; i < row; i++) {
                    int sum = 0;
                    for (int j = 0; j < col; j++) {
                        sum += matrix[i][j];
                        int cur = sum;
                        if (i > 0) {
                            cur += sumMatrix[i - 1][j];
                        }
                        sumMatrix[i][j] = cur;
                    }
                }
            }
        }

        //O(1)
        public int sumRegion(int row1, int col1, int row2, int col2) {
            int sum = this.sumMatrix[row2][col2];
            int flag = 0;
            if (row1 - 1 >= 0) {
                sum -= this.sumMatrix[row1 - 1][col2];
                flag++;
            }
            if (col1 - 1 >= 0) {
                sum -= this.sumMatrix[row2][col1 - 1];
                flag++;
            }
            if (flag == 2) {
                sum += this.sumMatrix[row1 - 1][col1 - 1];
            }
            return sum;
        }
    }

    /*
        308. Range Sum Query 2D - Mutable
     */
    static class NumMatrix2 {

        private int[][] sumMatrix;
        private int[][] matrix;

        public NumMatrix2(int[][] matrix) {
            this.matrix = matrix;
            int row = matrix.length;
            int col = row == 0 ? 0 : matrix[0].length;
            sumMatrix = new int[row][col];
            if (row != 0) {
                for (int i = 0; i < row; i++) {
                    int sum = 0;
                    for (int j = 0; j < col; j++) {
                        sum += matrix[i][j];
                        int cur = sum;
                        if (i > 0) {
                            cur += sumMatrix[i - 1][j];
                        }
                        sumMatrix[i][j] = cur;
                    }
                }
            }
        }

        public void update(int row, int col, int val) {
            int diff = val - matrix[row][col];
            matrix[row][col] = val;
            for (int i = row; i < sumMatrix.length; i++) {
                for (int j = col; j < sumMatrix[0].length; j++) {
                    sumMatrix[i][j] += diff;
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            int sum = this.sumMatrix[row2][col2];
            int flag = 0;
            if (row1 - 1 >= 0) {
                sum -= this.sumMatrix[row1 - 1][col2];
                flag++;
            }
            if (col1 - 1 >= 0) {
                sum -= this.sumMatrix[row2][col1 - 1];
                flag++;
            }
            if (flag == 2) {
                sum += this.sumMatrix[row1 - 1][col1 - 1];
            }
            return sum;
        }
    }

    /*
        lc 221. Maximal Square
        time O(N*M)
        Space o(1)
     */
    public int maximalSquare(char[][] matrix) {
        int row = matrix.length;
        if (row == 0) {
            return 0;
        }
        int res = checkMin(matrix);
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < matrix[0].length; j++) {
                int min = Math.min(matrix[i - 1][j] - '0', matrix[i][j - 1] - '0');
                min = Math.min(matrix[i - 1][j - 1] - '0', min);
                res = Math.max(res, min + 1);
                matrix[i][j] = (char) ('0' + min + 1);
            }
        }
        return res * res;
    }

    private int checkMin(char[][] matrix) {
        int row = matrix.length;
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == '1') {
                return 1;
            }
        }
        for (int i = 0; i < matrix[0].length; i++) {
            if (matrix[0][i] == '1') {
                return 1;
            }
        }
        return 0;
    }

    /*
        866. Prime Palindrome
           Time O(N) to find all palindrome
     */
    public int primePalindrome(int N) {
        //length of root, max is 5, because largest integer is 10 digits.
        for (int L = 1; L <= 5; ++L) {
            //Check for odd-length palindromes
            for (int root = (int) Math.pow(10, L - 1); root < (int) Math.pow(10, L); ++root) {
                StringBuilder sb = new StringBuilder(Integer.toString(root));
                for (int k = L - 2; k >= 0; --k) {
                    sb.append(sb.charAt(k));
                }
                int x = Integer.valueOf(sb.toString());
                if (x >= N && isPrime(x)) {
                    return x;
                }
                //If we didn't check for even-length palindromes:
                //return N <= 11 ? min(x, 11) : x
            }

            //Check for even-length palindromes
            for (int root = (int) Math.pow(10, L - 1); root < (int) Math.pow(10, L); ++root) {
                StringBuilder sb = new StringBuilder(Integer.toString(root));
                for (int k = L - 1; k >= 0; --k) {
                    sb.append(sb.charAt(k));
                }
                int x = Integer.valueOf(sb.toString());
                if (x >= N && isPrime(x)) {
                    return x;
                }
            }
        }

        throw null;
    }

    public boolean isPrime(int N) {
        if (N < 2) {
            return false;
        }
        int R = (int) Math.sqrt(N);
        for (int d = 2; d <= R; ++d) {
            if (N % d == 0) {
                return false;
            }
        }
        return true;
    }

    public int reverseDigit(int N) {
        int ans = 0;
        while (N > 0) {
            ans = 10 * ans + (N % 10);
            N /= 10;
        }
        return ans;
    }

    static double sqrt1(double n, double PRECISION) {
        double min, max; //min代表下边界，max代表上边界，mid为中间值也作为近似值
        min = 0;
        max = n;
        while (min + PRECISION < max) {
            double mid = (max + min) / 2;
            if (mid * mid < n + PRECISION) {
                min = mid; //值偏小，升高下边界
            }

            if (mid * mid > n - PRECISION) {
                max = mid;//值偏大，降低上边界
            }
        }
        return max;
    }

    public static int sqrt(int num) {
        if (num <= 0) {
            return 0;
        }
        int low = 0;
        int high = num / 2;
        while (low + 1 < high) {
            int mid = low + (high - low) / 2;
            int temp = num / mid;
            if (temp > mid) {
                low = mid;
            } else if (temp < mid) {
                high = mid;
            } else {
                return mid;
            }
        }
        if (high * high <= num) {
            return high;
        }
        return low;
    }

    /*
        97. Interleaving String
        time O(s1.length*s2.length)
        space O(s1.length*s2.length)
        这个方法的前提建立于：判断一个 s3s3 的前缀（用下标 kk 表示），能否用 s1s1 和 s2s2 的前缀（下标分别为 ii 和 jj），
        仅仅依赖于 s1 前 i 个字符和 s2 前 j 个字符，而与后面的字符无关。为了实现这个算法， 我们将使用一个 2D 的布尔数组 dp。
        dp[i][j] 表示用 s1 的前 (i+1) 和 s2 的前 (j+1) 个字符，总共 (i+j+2)个字符，是否交错构成 s3 的前缀。为了求出 dp[i][j]，
        我们需要考虑 2 种情况：s1 的第 i 个字符和 s2 的第 j 个字符都不能匹配 s3 的第 k 个字符，其中 k=i+j+1 。这种情况下，s1 和
        s2 的前缀无法交错形成 s3 长度为 k+1 的前缀。因此，我们让 dp[i][j] 为 False。s1 的第 i 个字符或者 s2 的第
        j 个字符可以匹配 s3 的第 k 个字符，其中 k=i+j+1 。假设匹配的字符是 x 且与 s1 的第 i 个字符匹配，我们就需要把 x
        放在已经形成的交错字符串的最后一个位置。此时，为了我们必须确保 s1 的前 ((i−1) 个字符和 s2 的前 j个字符能形成 s3 的一个前缀。
        类似的，如果我们将 s2 的第 j个字符与 s3 的第 k 个字符匹配，我们需要确保 s1的前 i 个字符和 s2 的前 (j−1) 个字符能形成s3 的一个前缀，
        我们就让 dp[i][j] 为 TrueTrue 。
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s3.length() != s1.length() + s2.length()) {
            return false;
        }
        boolean dp[][] = new boolean[s1.length() + 1][s2.length() + 1];
        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = true;
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1);
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1);
                } else {
                    dp[i][j] =
                        (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) || (dp[i][j - 1]
                            && s2.charAt(j - 1) == s3.charAt(i + j - 1));
                }
            }
        }
        return dp[s1.length()][s2.length()];
    }


    /*
        57. Insert Interval
        Time O(N) one pass to form result
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> s = new ArrayList<>(Arrays.asList(intervals));

        int i = 0;
        while (i < s.size() && !canMerge(newInterval, s.get(i))) {
            if (s.get(i)[0] > newInterval[1]) {
                s.add(i, newInterval);
                return listToArray(s);
            }
            i++;
        }
        if (i == s.size()) {
            s.add(newInterval);
            return listToArray(s);
        }
        //merge the current one with new interval
        s.set(i, new int[]{Math.min(newInterval[0], s.get(i)[0]),
            Math.max(newInterval[1], s.get(i)[1])});

        //merge the rest list.
        for (; i < s.size() - 1; i++) {
            int j = i + 1;
            if (canMerge(s.get(i), s.get(j))) {
                int[] newArray = {s.get(i)[0],
                    s.get(j)[1] > s.get(i)[1] ? s.get(j)[1] : s.get(i)[1]};
                s.set(i, newArray);
                s.remove(s.get(j));
                i = i - 1;
            }
        }
        return listToArray(s);
    }

    private int[][] listToArray(List<int[]> s) {
        int[][] result = new int[s.size()][2];
        for (int i = 0; i < result.length; i++) {
            result[i] = s.get(i);
        }
        return result;
    }

    /*
        673. Number of Longest Increasing Subsequence
        time O(n^2)
     */
    public int findNumberOfLIS(int[] nums) {
        int N = nums.length;
        if (N <= 1) {
            return N;
        }
        int[] lengths = new int[N]; //lengths[i] = length of longest ending in nums[i]
        int[] counts = new int[N]; //count[i] = number of longest ending in nums[i]
        Arrays.fill(counts, 1);
        int maxLength = 0;
        int ans = 0;
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < j; ++i) {
                if (nums[i] < nums[j]) {
                    if (lengths[i] >= lengths[j]) {
                        lengths[j] = lengths[i] + 1;
                        counts[j] = counts[i];
                        maxLength = Math.max(maxLength, lengths[j]);
                    } else if (lengths[i] + 1 == lengths[j]) {
                        counts[j] += counts[i];
                    }

                }
            }
        }
        for (int i = 0; i < N; ++i) {
            if (lengths[i] == maxLength) {
                ans += counts[i];
            }
        }
        return ans;
    }

    /*
        300. Longest Increasing Subsequence
        Time O(N^2) dp
        Time O(N*logN) binary search with suffix array
     */


    public int lengthOfLISBS(int[] nums) {
        int[] dp = new int[nums.length];
        int maxLen = 0;
        for (int num : nums) {
            int i = 0;
            int j = maxLen;
            while (i < j) {
                int m = (i + j) / 2;
                if (num > dp[m]) {
                    i = m + 1;
                } else {
                    j = m;
                }
            }
            dp[i] = num;
            maxLen = Math.max(i + 1, maxLen);
        }
        return maxLen;
    }


    //dp[i] means the longest increased sequence with nums[i]
    public int lengthOfLISDP(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int maxans = 1;
        for (int i = 1; i < dp.length; i++) {
            int maxval = 0;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    maxval = Math.max(maxval, dp[j]);
                }
            }
            dp[i] = maxval + 1;
            maxans = Math.max(maxans, dp[i]);
        }
        return maxans;
    }


    /*
        30. Substring with Concatenation of All Words
        Time O(s.length() * wordLengthSum)
        Space O(N)
     */
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        if (words.length == 0) {
            return res;
        }
        Map<String, Integer> count = new HashMap<>();
        for (String word : words) {
            count.put(word, count.getOrDefault(word, 0) + 1);
        }
        int singleLength = words[0].length();
        int wordLengthSum = words.length * singleLength;
        int start = 0;
        outer:
        while (start <= s.length() - wordLengthSum) {
            Map<String, Integer> countCopy = new HashMap<>();
            countCopy.putAll(count);
            int i = start;
            while (i - start < wordLengthSum) {
                String curWord = s.substring(i, i + singleLength);
                int curWordFreq = countCopy.getOrDefault(curWord, 0);
                if (curWordFreq > 0) {
                    i = i + singleLength;
                    countCopy.put(curWord, curWordFreq - 1);
                } else {
                    start++;
                    continue outer;
                }
            }
            res.add(start);
            start++;
        }
        return res;
    }

    /*
        翻转 k组链表
        lc 25. Reverse Nodes in k-Group
        time O(N) length of the list
        space O(1)
     */

    public ListNode reverseKGroup(ListNode head, int k) {
        if (k <= 1) {
            return head;
        }
        int length = getLengthOfALinkedList(head);
        if (length == 1 || k > length) {
            return head;
        }
        ListNode prev = null, cur = head, next = null;
        int groups = length / k;
        //mark the result's head.
        ListNode resHead = null;
        //mark each groups end, to link each group after reverse.
        ListNode prevGroupEnd = null, curGroupEnd = null;
        for (int t = 0; t < groups; t++) {
            //moving groups
            prevGroupEnd = curGroupEnd;
            curGroupEnd = cur;
            //reverse k times
            for (int i = 0; i < k; i++) {
                next = cur.next;
                cur.next = prev;
                prev = cur;
                cur = next;
            }
            if (resHead == null) {
                resHead = prev;
            }
            //link prevGroupEnding element to this group's first one.
            if (prevGroupEnd != null) {
                prevGroupEnd.next = prev;
            }
        }
        //link the last group to the remains next elements. if there is no more, link it to null.
        curGroupEnd.next = cur;
        return resHead;
    }

    private int getLengthOfALinkedList(ListNode head) {
        int count = 0;
        while (head != null) {
            count++;
            head = head.next;
        }
        return count;
    }

    /*
        239. Sliding Window Maximum
        Time O(klogk) to init heap, O(nk) to slide the window.
        Space O( max(k,nums.length-k+1))

        opt to O(N)
        The algorithm is quite straight forward :
        Process the first k elements separately to initiate the deque.
        Iterate over the array. At each step :
        Clean the deque :
            Keep only the indexes of elements from the current sliding window.
            Remove indexes of all elements smaller than the current one, since they will not be the maximum ones.
        Append the current element to the deque.
        Append deque[0] to the output.
        Return the output array.
        第一次把前k个的index加入队列
        每往后滑动时，将不在window里的element移除
        并比较此刻队列中的所有

     */

    public int[] maxSlidingWindowOpt(int[] nums, int k) {
        int n = nums.length;
        if (n * k == 0) {
            return new int[0];
        }
        if (k == 1) {
            return nums;
        }
        Deque<Integer> deq = new ArrayDeque<>();
        for (int i = 0; i < k; i++) {
            clean_deque(i, k, deq, nums);
            deq.addLast(i);
        }
        int[] output = new int[n - k + 1];
        output[0] = nums[deq.getFirst()];

        // build output
        for (int i = k; i < n; i++) {
            clean_deque(i, k, deq, nums);
            deq.addLast(i);
            output[i - k + 1] = nums[deq.getFirst()];
        }
        return output;
    }

    public void clean_deque(int i, int k, Deque<Integer> deq, int[] nums) {
        // remove indexes of elements not from sliding window
        if (!deq.isEmpty() && deq.getFirst() == i - k) {
            deq.removeFirst();
        }

        // remove from deq indexes of all elements
        // which are smaller than current element nums[i]
        //可先采用遍历数组方法删除所有小于当前与元素的序号
        //opt：从后往前删，因为从前到后是递减的，这样子不会删漏，也可不用遍历整个数组

        while (!deq.isEmpty() && nums[i] > nums[deq.getLast()]) {
            deq.removeLast();
        }
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        //if nums is empty
        if (nums.length == 0) {
            return new int[]{};
        }
        //if window is larger than nums.length
        if (nums.length < k) {
            k = nums.length;
        }
        PriorityQueue<Integer> window = new PriorityQueue<>(
            Comparator.comparingInt(o -> -o));
        for (int i = 0; i < k; i++) {
            window.add(nums[i]);
        }
        int[] res = new int[nums.length - k + 1];
        for (int i = 0; i < res.length; i++) {
            res[i] = window.peek();
            if (i < nums.length - k) {
                window.remove(nums[i]);
                window.add(nums[i + k]);
            }
        }
        return res;
    }


    /*
    692. Top K Frequent Words
    time O(Nlogk)
    space O(N)
     */
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> count = new HashMap();
        for (String word : words) {
            count.put(word, count.getOrDefault(word, 0) + 1);
        }
        PriorityQueue<String> heap = new PriorityQueue<String>(
            (w1, w2) -> count.get(w1).equals(count.get(w2)) ?
                w2.compareTo(w1) : count.get(w1) - count.get(w2));
        for (String word : count.keySet()) {
            heap.offer(word);
            if (heap.size() > k) {
                heap.poll();
            }
        }

        List<String> ans = new ArrayList();
        while (!heap.isEmpty()) {
            ans.add(heap.poll());
        }
        Collections.reverse(ans);
        return ans;
    }

    /*
    lc 347 347. Top K Frequent Elements
    Time complexity : O(Nlog(k)). The complexity of Counter method is
     O(N). To build a heap and output list takes O(Nlog(k)). Hence the overall complexity
      of the algorithm is O(N+Nlog(k))=O(Nlog(k)).

    Space complexity : O(N) to store the hash map.
     */
    public List<Integer> topKFrequent(int[] nums, int k) {
        // build hash map : character and how often it appears
        HashMap<Integer, Integer> count = new HashMap();
        for (int n : nums) {
            count.put(n, count.getOrDefault(n, 0) + 1);
        }

        // init heap 'the less frequent element first'
        PriorityQueue<Integer> heap =
            new PriorityQueue<Integer>((n1, n2) -> count.get(n1) - count.get(n2));
        // keep k top frequent elements in the heap
        for (int n : count.keySet()) {
            heap.add(n);
            if (heap.size() > k) {
                heap.poll();
            }
        }

        // build output list
        List<Integer> top_k = new LinkedList();
        while (!heap.isEmpty()) {
            top_k.add(heap.poll());
        }
        Collections.reverse(top_k);
        return top_k;
    }








    /*
937. Reorder Data in Log Files
Time O(NlogN)
Space O(N)
     */

    public String[] reorderLogFiles(String[] logs) {
        Arrays.sort(logs, (log1, log2) -> {
            String[] split1 = log1.split(" ", 2);
            String[] split2 = log2.split(" ", 2);
            boolean isDigit1 = Character.isDigit(split1[1].charAt(0));
            boolean isDigit2 = Character.isDigit(split2[1].charAt(0));
            if (!isDigit1 && !isDigit2) {
                int cmp = split1[1].compareTo(split2[1]);
                if (cmp != 0) {
                    return cmp;
                }
                return split1[0].compareTo(split2[0]);
            }
            return isDigit1 ? (isDigit2 ? 0 : 1) : -1;
        });
        return logs;
    }

    /*
        505. The Maze II
        shortest distance to destination

        Time complexity : O(m∗n∗max(m,n)).
        Complete traversal of maze will be done in the worst case.
        Here, mm and nn refers to the number of rows and columns of the maze.
        Further, for every current node chosen, we can travel up to a maximum
        depth of max(m,n) in any direction.

        Space complexity :O(mn). distance array of size m*nm∗n is used.
     */
    private int[][] distance;

    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        if (maze.length == 0) {
            return -1;
        }
        distance = new int[maze.length][maze[0].length];
        for (int[] d : distance) {
            Arrays.fill(d, -1);
        }
        dfs(maze, start, destination, 0);
        return distance[destination[0]][destination[1]];
    }

    private void dfs(int[][] maze, int[] cur, int[] dest, int count) {
        if (distance[cur[0]][cur[1]] != -1 && distance[cur[0]][cur[1]] <= count) {
            return;
        }
        distance[cur[0]][cur[1]] = count;
        if (cur[0] == dest[0] && cur[1] == dest[1]) {
            return;
        }
        for (int i = 0; i < 4; i++) {
            int x = cur[0];
            int y = cur[1];
            int moveRow = rowOffset[i];
            int moveCol = colOffset[i];
            int moveCount = count;
            while (inMaze(x + moveRow, y + moveCol, maze) && maze[x + moveRow][y + moveCol] == 0) {
                x += moveRow;
                y += moveCol;
                moveCount++;
            }
            if (distance[x][y] != -1 && distance[x][y] <= moveCount) {
                continue;
            }
            dfs(maze, new int[]{x, y}, dest, moveCount);
        }
    }

    /*
    490 the maze
    Time complexity : O(mn). Complete traversal of maze will be done in the worst case.
    Here, mm and nn refers to the number of rows and coloumns of the maze.
    Space complexity : O(mn). visited array of size
    m∗n is used and m*n stack used by dfs.
     */

    private int[] rowOffset = new int[]{1, 0, -1, 0};
    private int[] colOffset = new int[]{0, -1, 0, 1};
    private boolean[][] isVisited;

    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        if (maze.length == 0) {
            return false;
        }
        isVisited = new boolean[maze.length][maze[0].length];
        return dfs(maze, start, destination);
    }

    private boolean dfs(int[][] maze, int[] cur, int[] dest) {
        if (isVisited[cur[0]][cur[1]] == true) {
            return false;
        }
        isVisited[cur[0]][cur[1]] = true;
        if (cur[0] == dest[0] && cur[1] == dest[1]) {
            return true;
        }
        boolean flag = false;
        for (int i = 0; i < 4; i++) {
            int x = cur[0];
            int y = cur[1];
            int moveRow = rowOffset[i];
            int moveCol = colOffset[i];
            while (inMaze(x + moveRow, y + moveCol, maze) && maze[x + moveRow][y + moveCol] == 0) {
                x += moveRow;
                y += moveCol;
            }
            flag = dfs(maze, new int[]{x, y}, dest);
            if (flag) {
                return true;
            }
        }
        return false;
    }

    private boolean inMaze(int x, int y, int[][] maze) {
        return x >= 0 && x < maze.length && y >= 0 && y < maze[0].length;
    }

    /*
    lc 149 max points on a line
    Time O(N^2) enum each line based on two points which is N-1+N-2+...+2+1 ~~= N^2
    Space O(N)  to track down not more than N - 1 lines.
     */

    public int maxPoints(int[][] points) {
        if (points.length < 3) {
            return points.length;
        }
        int max = 2;
        for (int i = 0; i < points.length - 1; i++) {
            max = Math.max(max, pointsCountOnlinesThatPassOnePoint(points, i));
        }
        return max;
    }

    // i means index of current point.
    //precision is a problem when using double to calculate slope
    private int pointsCountOnlinesThatPassOnePoint(int[][] points, int i) {
        int max = 1;
        int samePoint = 0;
        //slope -> count
        Map<String, Integer> count = new HashMap<>();
        for (int j = i + 1; j < points.length; j++) {
            int x1 = points[i][0];
            int y1 = points[i][1];
            int x2 = points[j][0];
            int y2 = points[j][1];
            if (x1 == x2 && y1 == y2) {
                samePoint++;
                continue;
            }
            String slope;
            //represents horizontal line
            if (y1 == y2) {
                slope = "0";
            }
            //represents vertical line
            else if (x1 == x2) {
                slope = "Infinity";
            } else {
                int x = x1 - x2;
                int y = y1 - y2;
                int gcd = gcd(x, y);
                //gcd should be same sign as x
                x /= gcd;
                y /= gcd;
                //so x would always positive. y would represents positive or negative
                slope = Integer.toString(y) + "/" + Integer.toString(x);
            }
            count.put(slope, count.getOrDefault(slope, 1) + 1);
            max = Math.max(max, count.get(slope));
        }
        return max + samePoint;
    }

    //辗转相除法 最大公因数 greatest common divisor
    int gcd(int a, int b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }




/*
lc 721 accounts merge
another solution union find. assign a id for each email
 */

    public List<List<String>> accountsMergeDFS(List<List<String>> accounts) {
        Map<String, String> emailToName = new HashMap();
        Map<String, ArrayList<String>> graph = new HashMap();
        for (List<String> account : accounts) {
            String name = "";
            for (String email : account) {
                if (name == "") {
                    name = email;
                    continue;
                }
                graph.computeIfAbsent(email, x -> new ArrayList<String>()).add(account.get(1));
                graph.computeIfAbsent(account.get(1), x -> new ArrayList<String>()).add(email);
                emailToName.put(email, name);
            }
        }

        Set<String> seen = new HashSet();
        List<List<String>> ans = new ArrayList();
        for (String email : graph.keySet()) {
            if (!seen.contains(email)) {
                seen.add(email);
                Stack<String> stack = new Stack();
                stack.push(email);
                List<String> component = new ArrayList();
                while (!stack.empty()) {
                    String node = stack.pop();
                    component.add(node);
                    for (String nei : graph.get(node)) {
                        if (!seen.contains(nei)) {
                            seen.add(nei);
                            stack.push(nei);
                        }
                    }
                }
                Collections.sort(component);
                component.add(0, emailToName.get(email));
                ans.add(component);
            }
        }
        return ans;
    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, List<String>> accountDict = new HashMap<>();
        //for each account
        for (List<String> account : accounts) {
            //for each email address
            for (int i = 1; i < account.size(); i++) {
                String email = account.get(i);
                //find duplicate email
                if (accountDict.containsKey(email)) {
                    mergeTwoAccounts(account, accountDict.get(email), accountDict);
                    break;
                } else {
                    accountDict.put(email, account);
                }
            }
        }
        Set<List<String>> res = new HashSet<>();
        for (List<String> mergedAccount : new HashSet<>(accountDict.values())) {
            String name = mergedAccount.get(0);
            mergedAccount.remove(0);
            mergedAccount = new ArrayList<>(new HashSet<>(mergedAccount));
            Collections.sort(mergedAccount);
            mergedAccount.add(0, name);
            res.add(mergedAccount);
        }
        return new ArrayList<>(res);
    }

    //move emails from account1 to account2
    private void mergeTwoAccounts(List<String> account1, List<String> account2,
        Map<String, List<String>> accountDict) {
        if (account1.size() > account2.size()) {
            mergeTwoAccounts(account2, account1, accountDict);
            return;
        }
        for (int i = 1; i < account1.size(); i++) {
            String email = account1.get(i);
            account2.add(account1.get(i));
            accountDict.put(email, account2);

        }
    }


    /*
    lc 133 clone graph
     */

    /*
// Definition for a Tile.
class Tile {
    public int val;
    public List<Tile> neighbors;

    public Tile() {}

    public Tile(int _val,List<Tile> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/

    /*
        133. Clone Graph
        Time V+E
        Space N+E
     */
    Map<Node, Node> dict = new HashMap<>();

    public Node cloneGraph(Node node) {
        if (node == null) {
            return null;
        }
        if (dict.containsKey(node)) {
            return dict.get(node);
        }
        Node newNode = new Node(node.val);
        dict.put(node, newNode);
        for (Node neighbor : node.neighbors) {
            if (dict.containsKey(neighbor)) {
                newNode.neighbors.add(dict.get(neighbor));
            } else {
                newNode.neighbors.add(cloneGraph(neighbor));
            }
        }

        return newNode;
    }

    /*
    lc 438
    Time O(n) two pointer sliding window two pass.
     */
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        if (s.length() == 0) {
            return res;
        }
        //can speed up by using int[] to do counting
        Map<Character, Integer> pCount = new HashMap<>();
        for (char c : p.toCharArray()) {
            pCount.put(c, pCount.getOrDefault(c, 0) + 1);
        }
        Map<Character, Integer> curCount = new HashMap<>();
        //make a deep copy of pCount
        curCount.putAll(pCount);
        int i = 0;
        int j = i;
        while (j < s.length()) {
            char curChar = s.charAt(j);
            if (curCount.containsKey(curChar)) {
                int curCharCount = curCount.get(curChar);
                if (curCharCount == 1) {
                    curCount.remove(curChar);
                } else {
                    curCount.put(curChar, curCharCount - 1);
                }
                j++;
            } else {
                //check if this letter original needed
                if (pCount.containsKey(curChar)) {
                    curCount.put(s.charAt(i), curCount.getOrDefault(s.charAt(i), 0) + 1);
                    i++;
                } else {
                    j++;
                    i = j;
                    curCount = new HashMap<>();
                    //reset count dict.
                    curCount.putAll(pCount);
                }
            }
            if (curCount.size() == 0) {
                res.add(i);
                curCount.put(s.charAt(i), curCount.getOrDefault(s.charAt(i), 0) + 1);
                i++;
            }
        }
        return res;
    }

    interface HtmlParser {

        List<String> getUrls(String url);

    }

    //lc1236 web crawler
    public List<String> crawl(String startUrl, HtmlParser htmlParser) {
        Set<String> set = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        String hostname = getHostname(startUrl);

        queue.offer(startUrl);
        set.add(startUrl);

        while (!queue.isEmpty()) {
            String currentUrl = queue.poll();
            for (String url : htmlParser.getUrls(currentUrl)) {
                if (hostname.equals(getHostname(url)) && !set.contains(url)) {
                    queue.offer(url);
                    set.add(url);
                }
            }
        }

        return new ArrayList<String>(set);
    }

    private String getHostname(String url) {
        return url.substring(url.indexOf("//") + 2).split("/")[0];
    }

    /*
    159. Longest Substring with At Most Two Distinct Characters
    Example 1:

    Input: "eceba"
    Output: 3
    Explanation: t is "ece" which its length is 3.
    Example 2:

    Input: "ccaabbb"
    Output: 5
    Explanation: t is "aabbb" which its length is 5.
    Time complexity : O(N) where N is a number of characters in the input string.
    Space complexity : O(1) since additional space is used only for a hashmap with at most 3 elements.
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        Map<Character, Integer> dict = new HashMap<>();
        int l = 0, cur = 0;
        int res = 0;
        while (cur < s.length()) {
            char c = s.charAt(cur);
            if (dict.containsKey(c)) {
                dict.put(c, dict.get(c) + 1);
            } else {
                while (l <= cur && dict.size() == 2) {
                    char leftChar = s.charAt(l);
                    dict.put(leftChar, dict.get(leftChar) - 1);
                    if (dict.get(leftChar) == 0) {
                        dict.remove(leftChar);
                    }
                    l++;
                }
                dict.put(c, 1);
            }
            res = Math.max(res, cur - l + 1);
            cur++;
        }
        return res;
    }

    /*
    340. Longest Substring with At Most K Distinct Characters
    Time complexity : O(N) where N is a number of characters in the input string.
    Space complexity : O(k) since additional space is used only for a hashmap with at most k elements.
     */
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        if (k <= 0) {
            return 0;
        }
        Map<Character, Integer> dict = new HashMap<>();
        int l = 0, cur = 0;
        int res = 0;
        while (cur < s.length()) {
            char c = s.charAt(cur);
            if (dict.containsKey(c)) {
                dict.put(c, dict.get(c) + 1);
            } else {
                while (l <= cur && dict.size() == k) {
                    char leftChar = s.charAt(l);
                    dict.put(leftChar, dict.get(leftChar) - 1);
                    if (dict.get(leftChar) == 0) {
                        dict.remove(leftChar);
                    }
                    l++;
                }
                dict.put(c, 1);
            }
            res = Math.max(res, cur - l + 1);
            cur++;
        }
        return res;
    }

    /*
    76. Minimum Window Substring
    Time complexity O(s.length+t.length) calculate chars count in t. and then sliding window in s. both one pass
       space complexity O(t.length)
     */
    public String minWindow(String s, String t) {
        if (s.length() == 0 || t.length() == 0 || t.length() > s.length()) {
            return "";
        }
        Map<Character, Integer> charCount = new HashMap<>();
        for (char c : t.toCharArray()) {
            charCount.put(c, charCount.getOrDefault(c, 0) + 1);
        }
        int cur = 0, start = 0;
        //remain means how many letters still needed
        int remain = charCount.size();
        //stored the min windows's start index and end index
        int[] ans = new int[2];
        ans[1] = s.length() - 1;
        while (cur < s.length()) {
            char curChar = s.charAt(cur);
            if (charCount.containsKey(curChar)) {
                //if there is a letter we need, update the count dict with -1, which means we need 1 less this letter.
                charCount.put(curChar, charCount.get(curChar) - 1);
                //when it reach 0. means we need no more this letter.
                //remain--, means one less letter needed.
                if (charCount.get(curChar) == 0) {
                    remain--;
                }
            }
            //no more letters needed, is potential valid answer.
            if (remain == 0) {
                while (start <= cur) {
                    //if now have better solution, store it.
                    if (cur - start < ans[1] - ans[0]) {
                        ans[0] = start;
                        ans[1] = cur;
                    }
                    char left = s.charAt(start);
                    if (charCount.containsKey(left)) {
                        //can't move to right, current left letter is needed
                        if (charCount.get(left) == 0) {
                            break;
                        }
                        //update the count
                        charCount.put(left, charCount.get(left) + 1);
                    }
                    //keep sliding to right
                    start++;
                }
            }
            cur++;
        }
        //if remain!=0, then no solution
        return remain == 0 ? s.substring(ans[0], ans[1] + 1) : "";
    }

    //lc 191 count how many 1s in a integer
    public int hammingWeight(int n) {
        int bits = 0;
        int mask = 1;
        for (int i = 0; i < 32; i++) {
            if ((n & mask) != 0) {
                bits++;
            }
            mask <<= 1;
        }
        return bits;
    }

    /*
        lc75
        time O(N) one pass
        space O(1)
        While curr <= right :

        If nums[cur] = 0 : swap cur and left elements and move both pointers to the right.

        If nums[cur] = 2 : swap cur and right elements. Move pointer p2 to the left.

        If nums[cur] = 1 : move pointer cur to the right.
     */
    public void sortColors(int[] nums) {
        int cur = 0;
        //表示左边的0的边界
        int left = 0;
        //表示右边2的边界
        int right = nums.length - 1;
        while (cur <= right) {
            if (nums[cur] == 0) {
                swap(nums, left, cur);
                left++;
                cur++;
            } else if (nums[cur] == 2) {
                swap(nums, right, cur);
                right--;
            } else if (nums[cur] == 1) {
                cur++;
            }
        }
    }

    /*
    224. Basic Calculator
    1.中缀变后缀表达式
    2.后缀表达式用栈直接算：
    碰到数字入栈
    碰到符号 将栈顶两个数字拿出来运算后再入栈
     */

    //计算器
    public int calculate(String exp) {
        List<String> suffixExp = toSuffixExpression(exp);
        Stack<String> helper = new Stack<>();
        int res = 0;
        for (String c : suffixExp) {
            if (!isOperator(c.charAt(0))) {
                helper.push(c);
            } else if (!helper.isEmpty()) {
                int b;
                try {
                    b = Integer.parseInt(helper.pop());
                } catch (Exception e) {
                    b = Integer.MIN_VALUE;
                }

                int a = 0;
                if (helper.isEmpty()) {
                    a = 0;
                } else {
                    a = Integer.parseInt(helper.pop());
                }
                switch (c) {
                    case "+":
                        res = a + b;
                        break;
                    case "-":
                        res = a - b;
                        break;
                    case "*":
                        res = a * b;
                        break;
                    case "/":
                        res = a / b;
                        break;
                }

                helper.push(String.valueOf(res));
            }
        }
        return Integer.parseInt(helper.pop());
    }

    //计算器 中缀表达式转后缀表达式
    /*
    1.遇到操作数：直接添加到后缀表达式中
    2.栈为空时，遇到运算符，直接入栈
    3.遇到左括号：将其入栈
    4.遇到右括号：执行出栈操作，并将出栈的元素输出，直到弹出栈的是左括号，左括号不输出。
    5.遇到其他运算符：加减乘除：弹出所有优先级大于或者等于该运算符的栈顶元素，然后将该运算符入栈。
    6.最终将栈中的元素依次出栈，输出。
     */
    private List<String> toSuffixExpression(String exp) {
        Stack<Character> help = new Stack<>();
        List<String> res = new ArrayList<>();
        StringBuilder lastNumber = new StringBuilder();
        char lastValid = 0;
        for (int i = 0; i < exp.length(); i++) {
            if (exp.charAt(i) == ' ') {
                continue;
            }
            //是数字
            if (isNumber(exp.charAt(i))) {
                lastNumber.append(exp.charAt(i));
            } else if (lastNumber.length() > 0) {
                res.add(lastNumber.toString());
                lastNumber = new StringBuilder();
            }
            if (exp.charAt(i) == '(') {
                help.push(exp.charAt(i));
            } else if (exp.charAt(i) == ')') {
                while (help.peek() != '(') {
                    res.add(String.valueOf(help.pop()));
                }
                help.pop();
            } else if (isOperator(exp.charAt(i))) {
                if (i == 0 || (!isNumber(lastValid) && lastValid != ')')) {
                    res.add("0");
                }
                while (!help.isEmpty() && isOperator(help.peek()) && isPriority(help.peek(),
                    exp.charAt(i))) {
                    res.add(String.valueOf(help.pop()));
                }
                help.push(exp.charAt(i));
            }
            lastValid = exp.charAt(i);
        }
        if (lastNumber.length() > 0) {
            res.add(lastNumber.toString());
        }
        while (!help.isEmpty()) {
            res.add(String.valueOf(help.pop()));
        }
        return res;
    }

    private boolean isNumber(char c) {
        return c >= '0' && c <= '9';
    }

    private boolean isPriority(char a, char b) {

        return (a == '/' || a == '*') || (b == '+' || b == '-');
    }

    private boolean isOperator(char c) {
        return c == '+' || c == '*' || c == '-' || c == '/';
    }

    //计算器one pass
    int k = 0;

    public int calculateOnePass(String s) {
        char sign = '+';
        int curr = 0;
        int prev = 0;
        int num = 0;
        while (k < s.length()) {
            char ch = s.charAt(k);
            k++;
            if (ch == ' ') {
                continue;
            } else if (Character.isDigit(ch)) {
                num = num * 10 + ch - '0';
            } else if (ch == '(') {
                num = calculateOnePass(s);
            } else if (ch == ')') {
                break;
            } else {
                //先算乘除
                if (sign == '/') {
                    curr /= num;
                } else if (sign == '*') {
                    curr *= num;
                } else if (sign == '+') {
                    curr = num;
                } else {
                    curr = -num;
                }
                num = 0;
                sign = ch;
                //后算加减
                if (ch == '+' || ch == '-') {
                    prev += curr;
                    curr = 0;
                }
            }
        }
        if (sign == '/') {
            curr /= num;
        } else if (sign == '*') {
            curr *= num;
        } else if (sign == '+') {
            curr = num;
        } else {
            curr = -num;
        }
        return prev + curr;
    }

    /*
    二分查找法
    1.找出二维矩阵中最小的数left，最大的数right，那么第k小的数必定在left~right之间
    2.mid=(left+right) / 2；在二维矩阵中寻找小于等于mid的元素个数count
    3.若这个count小于k，表明第k小的数在右半部分且不包含mid，即left=mid+1, right=right，又保证了第k小的数在left~right之间
    4.若这个count大于k，表明第k小的数在左半部分且可能包含mid，即left=left, right=mid，又保证了第k小的数在left~right之间
    5.因为每次循环中都保证了第k小的数在left~right之间，当left==right时，第k小的数即被找出，等于right
     */

    public int kthSmallestBS(int[][] matrix, int k) {
        int lo = matrix[0][0], hi = matrix[matrix.length - 1][matrix[0].length - 1] + 1;//[lo, hi)
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int count = 0, j = matrix[0].length - 1;
            for (int i = 0; i < matrix.length; i++) {
                while (j >= 0 && matrix[i][j] > mid) {
                    j--;
                }
                count += (j + 1);
            }
            if (count < k) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    //time: O(min(n,k)*LogN) to build heap 使用堆
    public int kthSmallest(int[][] matrix, int k) {
        int length = matrix.length;
        if (length == 0) {
            return -1;
        }
        int width = matrix[0].length;
        PriorityQueue<Pair<Pair<Integer, Integer>, Integer>> smallestHeap = new PriorityQueue<Pair<Pair<Integer, Integer>, Integer>>(
            Comparator.comparingInt(Pair::getValue));
        for (int i = 0; i < width; i++) {
            smallestHeap.offer(new Pair(new Pair(0, i), matrix[0][i]));
        }
        for (int K = 0; K < k - 1; K++) {
            Pair<Integer, Integer> p = smallestHeap.poll().getKey();
            if (p.getKey() + 1 < length) {
                smallestHeap.offer(new Pair(new Pair(p.getKey() + 1, p.getValue()),
                    matrix[p.getKey() + 1][p.getValue()]));
            }
        }
        return smallestHeap.peek().getValue();
    }


    public int maxAreaOfIsland(int[][] grid) {
        int row = grid.length;
        if (row == 0) {
            return 0;
        }
        int col = grid[0].length;
        int max = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    max = Math.max(max, bfsMaxIsland(grid, i, j));
                }
            }
        }
        return max;
    }

    private int bfsMaxIsland(int[][] grid, int i, int j) {
        int count = 1;
        grid[i][j] = 0;
        for (int t = 0; t < 4; t++) {
            int newRow = i + rowOffset[t];
            int newCol = j + colOffset[t];
            if (inGrid(newRow, newCol, grid) && grid[newRow][newCol] == 1) {
                count += bfsMaxIsland(grid, newRow, newCol);
            }
        }
        return count;
    }

    /*
    lc 472 输出所有能被list里其他单词拼接而成的词
     */
    //time O(n(words)*m(ave letters))  space O(Letters Numbers)
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        Arrays.sort(words, Comparator.comparingInt(String::length));
        TrieNodeArray root = new TrieNodeArray();
        List<String> res = new ArrayList<>();
        for (String word : words) {
            TrieNodeArray node = root;
            for (Character c : word.toCharArray()) {
                if (node.children[c - 'a'] != null) {
                    node = node.children[c - 'a'];
                } else {
                    node.children[c - 'a'] = new TrieNodeArray();
                    node = node.children[c - 'a'];
                }
            }
            node.word = new String(word);

            int count = 0;
            TrieNodeArray curNode = root;
            for (char c : word.toCharArray()) {
                curNode = curNode.children[c - 'a'];
                count++;
                if (curNode != null && !word.equals(curNode.word)) {
                    if (curNode.word != null && checkIfConcatenated(root, word.substring(count))) {
                        res.add(word);
                        break;
                    }
                }
            }
        }
        return res;
    }


    private boolean checkIfConcatenated(TrieNodeArray root, String word) {
        if (word.length() == 0) {
            return true;
        }
        TrieNodeArray node = root;
        int count = 0;
        for (char c : word.toCharArray()) {
            if (node.children[c - 'a'] != null) {
                node = node.children[c - 'a'];
                count++;
                if (node.word != null) {
                    if (checkIfConcatenated(root, word.substring(count))) {
                        return true;
                    }
                }
            } else {
                return false;
            }
        }
        return false;
    }


    //用frist index做映射
    public boolean isIsomorphic1(String s, String t) {
        char[] ch1 = s.toCharArray();
        char[] ch2 = t.toCharArray();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            if (s.indexOf(ch1[i]) != t.indexOf(ch2[i])) {
                return false;
            }
        }
        return true;

    }

    /*
    lc205
    给定两个字符串 s 和 t，判断它们是否是同构的。

    如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。

    所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

     */
    public static boolean isIsomorphic(String s, String t) {
        Map<Character, Character> smap = new HashMap<>();
        Set<Character> hasAssign = new HashSet<>();
        for (int i = 0; i < s.length(); i++) {
            if (smap.containsKey(s.charAt(i)) && smap.get(s.charAt(i)) != t.charAt(i)) {
                return false;
            }
            if (!smap.containsKey(s.charAt(i)) && hasAssign.contains(t.charAt(i))) {
                return false;
            }
            smap.put(s.charAt(i), t.charAt(i));
            hasAssign.add(t.charAt(i));
        }
        return true;
    }


    public void solveSudoku(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        solve(board);
    }

    public boolean solve(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {//trial. Try 1 through 9
                        if (isValid(board, i, j, c)) {
                            board[i][j] = c; //Put c for this cell

                            if (solve(board)) {
                                return true; //If it's the solution return true
                            } else {
                                board[i][j] = '.'; //Otherwise go back
                            }
                        }
                    }

                    return false;
                }
            }
        }
        return true;
    }

    private boolean isValid(char[][] board, int row, int col, char c) {
        for (int i = 0; i < 9; i++) {
            if (board[i][col] == c) {
                return false; //check row
            }
            if (board[row][i] == c) {
                return false; //check column
            }
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] != '.' &&
                board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) {
                return false; //check 3*3 block
            }
        }
        return true;
    }


    public static boolean isValidSudoku(char[][] board) {
        Set<Character>[] row = new Set[9];
        Set<Character>[] col = new Set[9];
        Set<Character>[] box = new Set[9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                int boxIndex = (i / 3) * 3 + j / 3;
                if (row[i] == null) {
                    row[i] = new HashSet<>();
                }
                if (col[j] == null) {
                    col[j] = new HashSet<>();
                }
                if (box[boxIndex] == null) {
                    box[boxIndex] = new HashSet<>();
                }
                char cur = board[i][j];
                if (cur == '.') {
                    continue;
                }
                if (row[i].contains(cur) || col[j].contains(cur) || box[boxIndex]
                    .contains(cur)) {
                    return false;
                }
                row[i].add(cur);
                col[j].add(cur);
                box[boxIndex].add(cur);
            }
        }
        return true;
    }

    //类似，约瑟夫环，寻找递推公式 f(n个数，第m个删除) = [f(n-1,m)+m]%n  n>1
    //f(n,m) =1 n=1
    //通过删除一个数字之后重新开始排序，找到新的序列与原来的序列之间序号的关系

    //消除数字
    public int lastRemaining(int n) {
        //通过找规律分析f(n)与f(n/2)的递推公式
        //1 2 3 4 5 6 与 1 2 3 4 5 6 7 8 9 10 11 12
        //后者一次删除后变成 2 4 6 8 10 12
        //先除以二 变成1 2 3 4 5 6
        //第二次是倒过来删除，相当于 6 5 4 3 2 1
        //无论是否数组倒过来，最后剩下的位置的一定一样，这两个位置相加起来等于n/2+1
        //通过n/2+1-lastRemain(n/2) 求出最后的位置的数字，然后乘二恢复
        return n == 1 ? 1 : 2 * (n / 2 + 1 - lastRemaining(n / 2));
    }

    //循环链表删除节点
    static ListNode deleteNodeFromCircularList(ListNode head, int key) {
        if (head == null) {
            return null;
        }
        // Find the required node
        ListNode curr = head, prev = new ListNode(0);
        while (curr.val != key) {
            if (curr.next == head) {
                System.out.printf("\nGiven node is not found"
                    + " in the list!!!");
                break;
            }
            prev = curr;
            curr = curr.next;
        }

        // Check if node is only node
        if (curr.next == head) {
            head = null;
            return head;
        }
        // If more than one node, check if
        // it is first node
        if (curr == head) {
            prev = head;
            while (prev.next != head) {
                prev = prev.next;
            }
            head = curr.next;
            prev.next = head;
        }
        // check if node is last node
        else if (curr.next == head) {
            prev.next = head;
        } else {
            prev.next = curr.next;
        }
        return head;
    }

    //递归做法，每次判断截断字符串之后继续判断
    public int longestSubstring(String s, int k) {
        int len = s.length();
        if (len == 0 || k > len) {
            return 0;
        }
        if (k < 2) {
            return len;
        }

        return countCharater(s.toCharArray(), k, 0, len - 1);
    }

    private static int countCharater(char[] chars, int k, int p1, int p2) {
        if (p2 - p1 + 1 < k) {
            return 0;
        }
        int[] times = new int[26];  //  26个字母
        //  统计出现频次
        for (int i = p1; i <= p2; ++i) {
            ++times[chars[i] - 'a'];
        }
        //  如果该字符出现频次小于k，则不可能出现在结果子串中
        //  分别排除，然后挪动两个指针
        while (p2 - p1 + 1 >= k && times[chars[p1] - 'a'] < k) {
            ++p1;
        }
        while (p2 - p1 + 1 >= k && times[chars[p2] - 'a'] < k) {
            --p2;
        }

        if (p2 - p1 + 1 < k) {
            return 0;
        }
        //  得到临时子串，再递归处理
        for (int i = p1; i <= p2; ++i) {
            //  如果第i个不符合要求，切分成左右两段分别递归求得
            if (times[chars[i] - 'a'] < k) {
                return Math
                    .max(countCharater(chars, k, p1, i - 1), countCharater(chars, k, i + 1, p2));
            }
        }
        return p2 - p1 + 1;
    }


    /*
        416. Partition Equal Subset Sum
        method 1 和698一样回溯搜索所有可能
     */
    public boolean canPartition(int[] nums) {
        return canPartitionKSubsets(nums, 2);
    }

    /*平分成k组  回溯法暴力搜索 k parts
    lc698 Time Complexity:O(k^(N−k)*k!), where N is the length of nums,
    and kk is as given. As we skip additional zeroes in groups, naively we will make O(k!) calls to search,
    then an additional O(k^(N−k) ) calls after every element of groups is nonzero.
    Space Complexity: O(N)O(N), the space used by recursive calls to search in our call stack.
     */
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = Arrays.stream(nums).sum();
        if (sum % k > 0) {
            return false;
        }
        //必定为sum/k
        int target = sum / k;

        Arrays.sort(nums);
        //search from back to front
        int row = nums.length - 1;
        //prune the search space
        if (nums[row] > target) {
            return false;
        }
        while (row >= 0 && nums[row] == target) {
            row--;
            k--;
        }
        return searchPartitionK(new int[k], row, nums, target);
    }

    public boolean searchPartitionK(int[] groups, int row, int[] nums, int target) {
        //if each number had been placed, its successful.
        if (row < 0) {
            return true;
        }
        int v = nums[row--];
        for (int i = 0; i < groups.length; i++) {
            if (groups[i] + v <= target) {
                groups[i] += v;
                if (searchPartitionK(groups, row, nums, target)) {
                    return true;
                }
                groups[i] -= v;
            }
            //Because all numbers are positive, groups[i]==0 happens when there is a number can't fit into an empty group.
            //However, it also means the number can't be fitted into any group.
            //Therefore, there is no way to partition into k equal sum subsets.
            if (groups[i] == 0) {
                break;
            }
        }
        return false;
    }

    //2d array 找最低path到终点
    public int minPathSum(int[][] grid) {
        int row = grid.length;
        if (row == 0) {
            return 0;
        }
        int col = grid[0].length;
        int[][] dp = new int[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (i - 1 < 0 && j - 1 < 0) {
                    dp[i][j] = grid[i][j];
                } else if (i - 1 < 0 && j - 1 >= 0) {
                    dp[i][j] = dp[i][j - 1] + grid[i][j];
                } else if (i - 1 >= 0 && j - 1 < 0) {
                    dp[i][j] = dp[i - 1][j] + grid[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
                }
            }
        }
        return dp[row - 1][col - 1];
    }

    public int findCircleNum(int[][] M) {
        int N = M.length;
        if (N <= 1) {
            return N;
        }
        int[] root = new int[N];
        int[] ids = new int[N];
        for (int i = 0; i < N; i++) {
            root[i] = i;
            ids[i] = 1;
        }
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                if (M[i][j] == 1) {
                    union(i, j, root, ids);
                }
            }
        }
        return countSets(root);
    }

    /*
    983. Minimum Cost For Tickets
    Time O(N)天
     */
    public int mincostTickets(int[] days, int[] costs) {
        int[] dp = new int[days.length];
        int[] durations = new int[]{1, 7, 30};
        for (int i = dp.length - 1; i >= 0; i--) {
            int j = i;
            int min = Integer.MAX_VALUE;
            for (int d = 0; d < durations.length; d++) {
                //找到刚好此时票种无法覆盖到的最大的j
                while (j <= days.length - 1 && days[j] - days[i] < durations[d]) {
                    j++;
                }
                if (j == days.length) {
                    min = Math.min(min, costs[d]);
                } else {
                    //通过计算买了这个票种之后，到j天再买票的价格
                    min = Math.min(min, dp[j] + costs[d]);
                }
            }
            dp[i] = min;
        }
        return dp[0];
    }

    /*
        322. Coin Change
        Time O(amount*coins.length)
     */
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, -1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            int min = Integer.MAX_VALUE;
            boolean flag = false;
            for (int c : coins) {
                if (i - c >= 0 && dp[i - c] != -1) {
                    min = Math.min(min, dp[i - c] + 1);
                    flag = true;
                }
            }
            if (flag) {
                dp[i] = min;
            }
        }
        return dp[amount];
    }

    public int[][] intervalIntersection(int[][] A, int[][] B) {
        int L1 = A.length;
        int L2 = B.length;
        int i = 0, j = 0;
        List<int[]> res = new ArrayList<>();
        while (i < L1 && j < L2) {
            if (canMerge(A[i], B[j])) {
                int start = Math.max(A[i][0], B[j][0]);
                int end = Math.min(A[i][1], B[j][1]);
                res.add(new int[]{start, end});
            }
            if (A[i][1] > B[j][1]) {
                j++;
            } else {
                i++;
            }
        }
        int[][] resA = new int[res.size()][2];
        res.toArray(resA);
        return resA;
    }


    public List<Integer> arraysIntersection(int[] arr1, int[] arr2, int[] arr3) {
        List<Integer> res = new ArrayList<>();
        int L1 = arr1.length, L2 = arr2.length, L3 = arr3.length;
        int i = 0, j = 0, k = 0;
        while (i < L1 && j < L2 && k < L3) {
            if (arr1[i] == arr2[j] && arr1[i] == arr3[k]) {
                res.add(arr1[i]);
                i++;
                k++;
                j++;
            } else if (arr1[i] <= arr2[j] && arr1[i] <= arr3[k]) {
                i++;
            } else if (arr2[j] <= arr1[i] && arr2[j] <= arr3[k]) {
                j++;
            } else if (arr3[k] <= arr2[j] && arr3[k] <= arr1[i]) {
                k++;
            }
        }
        return res;
    }

    //两个数组 不去重，使用排序方法，或者hashmap计数
    public int[] intersect(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i = 0, j = 0, k = 0;
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] < nums2[j]) {
                ++i;
            } else if (nums1[i] > nums2[j]) {
                ++j;
            } else {
                nums1[k++] = nums1[i++];
                ++j;
            }

        }
        return Arrays.copyOfRange(nums1, 0, k);
    }

    //两个数组取交集， 要求去重
    public int[] intersection(int[] nums1, int[] nums2) {
        HashSet<Integer> set1 = new HashSet<Integer>();
        for (Integer n : nums1) {
            set1.add(n);
        }
        HashSet<Integer> set2 = new HashSet<Integer>();
        for (Integer n : nums2) {
            set2.add(n);
        }
        set1.retainAll(set2);
        int[] res = new int[set1.size()];
        int i = 0;
        for (int n : set1) {
            res[i++] = n;
        }

        return res;
    }

    public int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = (l + r) / 2;
            if (nums[mid] > nums[mid + 1]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    //using PQ
    public int firstMissingPositive1(int[] nums) {
        int min = 1;
        PriorityQueue<Integer> posNums = new PriorityQueue<>(Comparator.comparingInt(n -> n));
        for (int num : nums) {
            if (num > 0) {
                posNums.add(num);
            }
        }
        while (posNums.size() != 0) {
            if (min < posNums.peek()) {
                return min;
            } else if (min == posNums.peek()) {
                posNums.poll();
                min++;
            } else {
                posNums.poll();
            }
        }
        return min;
    }


    //use sign and index as hash
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;

        // Base case.
        int contains = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) {
                contains++;
                break;
            }
        }

        if (contains == 0) {
            return 1;
        }

        // nums = [1]
        if (n == 1) {
            return 2;
        }

        // Replace negative numbers, zeros,
        // and numbers larger than n by 1s.
        // After this convertion nums will contain
        // only positive numbers.
        for (int i = 0; i < n; i++) {
            if ((nums[i] <= 0) || (nums[i] > n)) {
                nums[i] = 1;
            }
        }

        // Use index as a hash key and number sign as a presence detector.
        // For example, if nums[1] is negative that means that number `1`
        // is present in the array.
        // If nums[2] is positive - number 2 is missing.
        for (int i = 0; i < n; i++) {
            int a = Math.abs(nums[i]);
            // If you meet number a in the array - change the sign of a-th element.
            // Be careful with duplicates : do it only once.
            if (a == n) {
                nums[0] = -Math.abs(nums[0]);
            } else {
                nums[a] = -Math.abs(nums[a]);
            }
        }

        // Now the index of the first positive number
        // is equal to first missing positive.
        for (int i = 1; i < n; i++) {
            if (nums[i] > 0) {
                return i;
            }
        }

        if (nums[0] > 0) {
            return n;
        }

        return n + 1;
    }


    /*
        202. Happy Number
     */
    public int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }

    public boolean isHappy(int n) {
        int slowRunner = n;
        int fastRunner = getNext(n);
        while (fastRunner != 1 && slowRunner != fastRunner) {
            slowRunner = getNext(slowRunner);
            fastRunner = getNext(getNext(fastRunner));
        }
        return fastRunner == 1;
    }

    Set<Integer> seen = new HashSet<>();

    public boolean isHappyRecursion(int n) {
        seen.add(n);
        int nextNum = 0;
        while (n != 0) {
            int currentDigit = n % 10;
            nextNum += currentDigit * currentDigit;
            n = n / 10;
        }
        if (nextNum == 1) {
            return true;
        }
        if (seen.contains(nextNum)) {
            return false;
        } else {
            return isHappyRecursion(nextNum);
        }

    }

    //Using stack
    public String removeDuplicateLetters2(String s) {

        Stack<Character> stack = new Stack<>();

        // this lets us keep track of what's in our solution in O(1) time
        HashSet<Character> seen = new HashSet<>();

        // this will let us know if there are any more instances of s[i] left in s
        HashMap<Character, Integer> last_occurrence = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            last_occurrence.put(s.charAt(i), i);
        }

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            // we can only try to add c if it's not already in our solution
            // this is to maintain only one of each character
            if (!seen.contains(c)) {
                // if the last letter in our solution:
                //     1. exists
                //     2. is greater than c so removing it will make the string smaller
                //     3. it's not the last occurrence
                // we remove it from the solution to keep the solution optimal
                while (!stack.isEmpty() && c < stack.peek()
                    && last_occurrence.get(stack.peek()) > i) {
                    seen.remove(stack.pop());
                }
                seen.add(c);
                stack.push(c);
            }
        }
        StringBuilder sb = new StringBuilder(stack.size());
        for (Character c : stack) {
            sb.append(c.charValue());
        }
        return sb.toString();
    }

    //字母去重后保留最小的字典排序
    public static String removeDuplicateLetters(String s) {
        if (s.length() <= 1) {
            return s;
        }
        int[] count = new int[26];
        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char cur = s.charAt(i);
            if (count[cur - 'a'] == 0) {
                continue;
            }
            char min = 'z';
            int j = i;
            int[] temp = Arrays.copyOf(count, 26);
            while (j < s.length() - 1 && count[cur - 'a'] != 1) {
                if (count[cur - 'a'] != 0) {
                    min = min < cur ? min : cur;

                    if (--temp[cur - 'a'] == 0) {
                        break;
                    }
                }
                j++;
                cur = s.charAt(j);
            }
            //The Last char or the char that must exist.
            min = min < cur ? min : cur;
            res.append(min);
            count[min - 'a'] = 0;
            //delete those before the char we want.
            for (char c : s.substring(i, s.indexOf(min, i)).toCharArray()) {
                if (count[c - 'a'] > 0) {
                    count[c - 'a']--;
                }
            }
            i = s.indexOf(min, i);
        }

        return res.toString();

    }

    //拓扑排序
    public int[] sortItems(int n, int m, int[] group, List<List<Integer>> beforeItems) {
        //构造小组的依赖关系和组内项目的依赖关系。
        //判断小组间是否有冲突，没冲突计算出小组的拓扑排序。
        //判断每个小组内项目是否有冲突，没冲突计算出小组内项目的拓扑排序。
        return null;
    }


    //topological sort 拓扑排序
    public int[] findOrder(int numCourses, int[][] prerequisites) {

        int[] inDegrees = new int[numCourses];
        for (int[] pre : prerequisites) {
            inDegrees[pre[0]]++;
        }
        Stack<Integer> readyToMove = new Stack<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegrees[i] == 0) {
                readyToMove.push(i);
            }
        }
        int count = 0;
        int[] res = new int[numCourses];
        while (!readyToMove.isEmpty()) {
            int movedCourse = readyToMove.pop();
            res[count] = movedCourse;
            count++;
            for (int[] pre : prerequisites) {
                if (pre[1] == movedCourse) {
                    if (--inDegrees[pre[0]] == 0) {
                        readyToMove.push(pre[0]);
                    }
                }
            }
        }
        return count == numCourses ? res : new int[]{};
    }

    /*
        207. Course Schedule
        Time v+e
        Space v or e
     */
    public static boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] inDegree = new int[numCourses];
        List<Integer>[] graph = new ArrayList[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] pre : prerequisites) {
            graph[pre[1]].add(pre[0]);
            inDegree[pre[0]]++;
        }
        int count = 0;
        Queue<Integer> readyToTake = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                readyToTake.offer(i);
            }
        }
        while (!readyToTake.isEmpty()) {
            int course = readyToTake.poll();
            count++;
            for (int link : graph[course]) {
                if (--inDegree[link] == 0) {
                    readyToTake.add(link);
                }
            }
        }
        return count == numCourses;


    }


    /*
    lc 733. Flood Fill
    dfs
    Time O(N) N is cell numbers
    Space O(N) max stack depth would be N
     */
    public static int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        if (image.length == 0) {
            return image;
        }
        fillFlood(image, sr, sc, image[sr][sc], newColor,
            new boolean[image.length][image[0].length]);
        return image;
    }

    //dfs
    private static void fillFlood(int[][] image, int sr, int sc, int oldColor, int newColor,
        boolean[][] visited) {
        int r = image.length;
        int c = image[0].length;
        if (sr >= 0 && sr < r && sc >= 0 && sc < c && !visited[sr][sc]) {
            if (image[sr][sc] == oldColor) {
                image[sr][sc] = newColor;
                visited[sr][sc] = true;
                fillFlood(image, sr + 1, sc, oldColor, newColor, visited);
                fillFlood(image, sr, sc + 1, oldColor, newColor, visited);
                fillFlood(image, sr - 1, sc, oldColor, newColor, visited);
                fillFlood(image, sr, sc - 1, oldColor, newColor, visited);
            } else {
                visited[sr][sc] = true;
            }
        }
    }

    //BF, check every possible subarray
    public int subarraySumBF(int[] nums, int k) {
        int count = 0;
        for (int start = 0; start < nums.length; start++) {
            int sum = 0;
            for (int end = start; end < nums.length; end++) {
                sum += nums[end];
                if (sum == k) {
                    count++;
                }
            }
        }
        return count;
    }

    public static int subarraySum(int[] nums, int k) {
        int count = 0, sum = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            //如果当前的累积和减去之前某一个累积和等于k，相当于中间有一段的和为k
            if (map.containsKey(sum - k)) {
                count += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }

        return count;
    }

    public static String smallestEquivalentString(String A, String B, String S) {
        int n = A.length();
        int[] root = new int[27];
        for (int i = 0; i < root.length; i++) {
            root[i] = i;
        }
        for (int i = 0; i < n; i++) {
            int a = A.charAt(i) - 'a';
            int b = B.charAt(i) - 'a';
            unionFindSmallDicString(b, a, root);

        }
        StringBuilder res = new StringBuilder();
        for (char c : S.toCharArray()) {
            char c1 = (char) (find(c - 'a', root) + 'a');
            res.append(c1);
        }
        return res.toString();
    }

    private static void unionFindSmallDicString(int a, int b, int[] root) {
        int aRoot = find(a, root);
        int bRoot = find(b, root);
        if (aRoot == bRoot) {
            return;
        }
        if (root[bRoot] > root[aRoot]) {
            root[bRoot] = root[aRoot];
        } else {
            root[aRoot] = root[bRoot];
        }
    }


    /*
    lc 23 merge k sorted list
    每次存入一个list的头部
    Time O(NLogK) N is number of all nodes
    Space O(k+N)
     */
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode head = new ListNode(0);
        ListNode cur = head;
        PriorityQueue<ListNode> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
        for (ListNode lh : lists) {
            if (lh != null) {
                pq.add(lh);
            }
        }
        while (pq.size() != 0) {
            ListNode min = pq.poll();
            cur.next = min;
            cur = cur.next;
            if (min.next != null) {
                pq.add(min.next);
            }
        }
        return head.next;
    }

    /*
        516. Longest Palindromic Subsequence
        Time(n^2)
     */
    public static int longestPalindromeSubseq(String s) {
        int length = s.length();
        int[][] dp = new int[length][length];
        for (int i = 0; i < length; i++) {
            dp[i][i] = 1;
        }
        for (int l = 1; l <= length; l++) {
            for (int i = 0; i < length - l; i++) {
                int j = i + l;
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][length - 1];
    }

    public int KMPindexOf(String source, String pattern) {
        int i = 0, j = 0;
        char[] src = source.toCharArray();
        char[] ptn = pattern.toCharArray();
        int sLen = src.length;
        int pLen = ptn.length;
        int[] next = getNext(ptn);
        while (i < sLen && j < pLen) {
            // 如果j = -1,或者当前字符匹配成功(src[i] = ptn[j]),都让i++,j++
            if (j == -1 || src[i] == ptn[j]) {
                i++;
                j++;
            } else {
                // 如果j!=-1且当前字符匹配失败,则令i不变,j=next[j],即让pattern模式串右移j-next[j]个单位
                j = next[j];
            }
        }
        if (j == pLen) {
            return i - j;
        }
        return -1;
    }

    protected int[] getNext(char[] p) {
        // 已知next[j] = k,利用递归的思想求出next[j+1]的值
        // 如果已知next[j] = k,如何求出next[j+1]呢?具体算法如下:
        // 1. 如果p[j] = p[k], 则next[j+1] = next[k] + 1;
        // 2. 如果p[j] != p[k], 则令k=next[k],如果此时p[j]==p[k],则next[j+1]=k+1,
        // 如果不相等,则继续递归前缀索引,令 k=next[k],继续判断,直至k=-1(即k=next[0])或者p[j]=p[k]为止
        int pLen = p.length;
        int[] next = new int[pLen];
        int k = -1;
        int j = 0;
        next[0] = -1; // next数组中next[0]为-1
        while (j < pLen - 1) {
            if (k == -1 || p[j] == p[k]) {
                k++;
                j++;
                // 修改next数组求法
                if (p[j] != p[k]) {
                    next[j] = k;// KMPStringMatcher中只有这一行
                } else {
                    // 不能出现p[j] = p[next[j]],所以如果出现这种情况则继续递归,如 k = next[k],
                    // k = next[[next[k]]
                    next[j] = next[k];
                }
            } else {
                k = next[k];
            }
        }
        return next;
    }


    /*
        42. Trapping Rain Water
        Time N
        Space N
        单调递减栈
     */
    public int trapWaterDecreaseStack(int[] height) {
        Stack<Integer> s = new Stack<>();
        int res = 0;//1
        for (int i = 0; i < height.length; i++) {
            while (s.size() > 1 && height[s.peek()] < height[i]) {

                int baseIndex = s.pop();
                res += (Math.min(height[s.peek()], height[i]) - height[baseIndex])
                    * (i - 1 - s.peek());
            }
            if (!s.isEmpty() && height[s.peek()] <= height[i]) {
                s.pop();
            }
            s.push(i);
        }
        return res;
    }

    /*
        two pointer

        Time N
        Space 1
        双指针一个最左一个最右边
        我们知道盛水是看左右最小的那个开始
        每次比较左右pointer 较小的那个
        维护一个从左边的目前最高 和从右边的目前最高
        该点能存的水就是 目前最高减去目前的高度
        每次从左右较矮的地方开始装水，确保该点一定是能装满水的
     */
    public int trapWaterTwoPointer(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        int res = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                leftMax = Math.max(leftMax, height[left]);
                res += leftMax - height[left];
                left++;
            } else {
                rightMax = Math.max(rightMax, height[right]);
                res += rightMax - height[right];
                right--;
            }
        }
        return res;
    }

    public static int waterPlants(int[] plants, int cap1, int cap2) {
        int n = plants.length;
        int count = 2;
        int cur1 = cap1;
        int cur2 = cap2;
        int i = 0;
        while (i <= n - 1 - i) {
            if (i == n - 1 - i) {
                if (cur1 + cur2 < plants[i]) {
                    count++;
                }
                break;
            } else {
                if (cur1 >= plants[i]) {
                    cur1 -= plants[i];
                } else {
                    count++;
                    cur1 = cap1;
                }
                if (cur2 >= plants[n - 1 - i]) {
                    cur2 -= plants[n - 1 - i];
                } else {
                    count++;
                    cur2 = cap2;
                }
                i++;
            }
        }

        return count;
    }

    public static int minDominoRotations(int[] A, int[] B) {
        int length = A.length;
        if (length == 0) {
            return -1;
        } else if (length == 1) {
            return 0;
        }
        int a = A[0];
        int b = B[0];
        int ua = 0;
        int ub = a == b ? -1 : 0;
        int la = 0;
        int lb = a == b ? -1 : 0;
        for (int i = 0; i < length; i++) {
            if (ua != -1) {
                if (A[i] != a && B[i] == a) {
                    ua++;
                } else if (A[i] != a && B[i] != a) {
                    ua = -1;
                    la = -1;
                }
            }

            if (la != -1) {
                if (B[i] != a && A[i] == a) {
                    la++;
                } else if (A[i] != a && B[i] != a) {
                    la = -1;
                    ua = -1;
                }
            }

            if (ub != -1) {
                if (B[i] == b && A[i] != b) {
                    ub++;
                } else if (A[i] != b && B[i] != b) {
                    ub = -1;
                    lb = -1;
                }
            }

            if (lb != -1) {
                if (B[i] != b && A[i] == b) {
                    lb++;
                } else if (A[i] != b && B[i] != b) {
                    lb = -1;
                    ub = -1;
                }
            }
            if (lb == -1 && ub == -1 && ua == -1 && la == -1) {
                return -1;
            }
        }

        int[] reses = new int[]{ua, la, ub, lb};
        int min = Integer.MAX_VALUE;
        for (int res : reses) {
            if (res != -1) {
                min = Math.min(res, min);
            }
        }
        return min;
    }

    /*
        724 Find Pivot Index
        using accumulative sum.
        for large Integer, we could have a big divisor, and then using quotient and remainder to
        calculate accumulative sum.
        Time O(N)
        Space O(1)
     */
    public static int pivotIndex(int[] nums) {
        int divisor = 1000000007;
        int quo1 = 0;
        int rema1 = 0;
        for (int num : nums) {
            quo1 += (num / divisor);
            rema1 += num % divisor;
            if (rema1 > divisor) {
                quo1 += rema1 / divisor;
                rema1 += rema1 % divisor;
            }
        }
        int quo2 = 0;
        int rema2 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i - 1 >= 0) {
                quo2 += nums[i - 1] / divisor;
                rema2 += nums[i - 1] % divisor;
            }
            if (rema2 > divisor) {
                quo2 += rema2 / divisor;
                rema2 += rema2 % divisor;
            }

            int quoCur = 0;
            quoCur += nums[i] / divisor;
            int remaCur = 0;
            remaCur += nums[i] % divisor;

            if (quo1 - quoCur - quo2 == quo2 && rema1 - remaCur - rema2 == rema2) {
                return i;
            }
        }
        return -1;
    }


    /*
    lc 973 K Closest Points to Origin
    1. using heap
    2. quick select

    for 1, Time O(NlogK)
    space O(k)
     */
    public int[][] kClosest(int[][] points, int K) {

        int[][] res = new int[K][2];
        PriorityQueue<int[]> queue = new PriorityQueue<>(K, (t1, t2) -> dis(t2) - dis(t1));//构建一个大根堆
        for (int i = 0; i < points.length; i++) {
            queue.offer(points[i]); //将所有元素入队
            if (queue.size() > K) {
                queue.poll();
            }
        }
        for (int i = 0; i < K; i++) {
            res[i] = queue.poll(); //前k个元素出堆，就是我们所需要的结果。
        }

        return res;
    }


    /*
    quick select solution
    Time avg O(N) worst O(N^2)
    Space O(N)
     */
    public int[][] kClosestQ(int[][] points, int K) {
        kClosedQuickS(points, 0, points.length - 1, K);
        return Arrays.copyOf(points, K);
    }

    public void kClosedQuickS(int[][] points, int low, int high, int K) {
        if (low < high) {
            int center = kClosedQuickP(points, low, high, K);
            if (center > K - 1) {
                kClosedQuickS(points, low, center - 1, K);
            } else {
                kClosedQuickS(points, center + 1, high, K);
            }

        }
    }

    public int kClosedQuickP(int[][] points, int low, int high, int K) {
        int[] pivot = points[low];
        while (low < high) {
            while (low < high && dis(pivot) < dis(points[high])) {
                high--;
            }
            if (low < high) {
                points[low] = points[high];
                low++;
            }
            while (low < high && dis(pivot) > dis(points[low])) {
                low++;
            }
            if (low < high) {
                points[high] = points[low];
                high--;
            }
        }
        points[low] = pivot;
        return low;
    }

    public int dis(int[] t)  //计算该点到原点的距离
    {
        return t[0] * t[0] + t[1] * t[1];
    }

    /*
        除法
        29. Divide Two Integers
     */
    public static int divide(int dividend, int divisor) {
        boolean sign = (dividend > 0) ^ (divisor > 0);
        int result = 0;
        if (dividend > 0) {
            dividend = -dividend;
        }
        if (divisor > 0) {
            divisor = -divisor;
        }
        while (dividend <= divisor) {
            int temp_result = -1;
            int temp_divisor = divisor;
            while (dividend <= (temp_divisor << 1)) {
                if (temp_divisor <= (Integer.MIN_VALUE >> 1)) {
                    break;
                }
                temp_result = temp_result << 1;
                temp_divisor = temp_divisor << 1;
            }
            dividend = dividend - temp_divisor;
            result += temp_result;
        }
        if (!sign) {
            if (result <= Integer.MIN_VALUE) {
                return Integer.MAX_VALUE;
            }
            result = -result;
        }
        return result;
    }


    /*
    lc 166 Fraction to Recurring Decimal


     */
    public String fractionToDecimal(int numerator, int denominator) {
        StringBuilder res = new StringBuilder();
        if (numerator == 0) {
            return "0";
        }
        //if denominator == 0

        //xor means different be true, same be false.
        if (numerator < 0 ^ denominator < 0) {
            res.append('-');
        }
        long n = Math.abs(Long.valueOf(numerator));
        long d = Math.abs(Long.valueOf(denominator));
        //quotient 商
        long quo = n / d;
        long remainder = n % d;
        if (remainder == 0) {
            res.append(quo);
            return res.toString();
        } else {
            res.append(quo);
            res.append('.');
            //use a hashmap to record each quotient's index, if there is a duplicate find,
            //then we found a recurrence. end loop, add a parentheses between  first appearance and the end.
            Map<Long, Integer> map = new HashMap<>();
            while (remainder != 0) {
                quo = remainder * 10 / d;
                if (map.containsKey(remainder)) {
                    res.insert(map.get(remainder), "(");
                    res.append(")");
                    break;
                }
                res.append(quo);
                //record this remainder which divide to current quotient.
                map.put(remainder, res.length() - 1);
                remainder = remainder * 10 % d;
            }
        }
        return res.toString();
    }

    public void reverseString(char[] s) {
        int left = 0, right = s.length - 1;
        while (left < right) {
            char tmp = s[left];
            s[left++] = s[right];
            s[right--] = tmp;
        }
    }

    public static List<Integer> connectedCities(int n, int g, List<Integer> originCities,
        List<Integer> destinationCities) {
        int[] root = new int[n + 1];
        int[] ids = new int[n + 1];

        for (int i = 0; i <= n; i++) {
            root[i] = i;
            ids[i] = 1;
        }

        for (int i = g + 1; i <= n; i++) {
            for (int j = 2 * i; j <= n; j += i) {
                union(j, i, root, ids);
            }
        }

        List<Integer> res = new ArrayList<>(originCities.size());
        Iterator<Integer> itSrc = originCities.iterator();
        Iterator<Integer> itDest = destinationCities.iterator();

        while (itSrc.hasNext() && itDest.hasNext()) {
            res.add(find(itSrc.next(), root) == find(itDest.next(), root) ? 1 : 0);
        }

        return res;
    }


    /*
        128. Longest Consecutive Sequence
        union find solution
        time O(N) nearly O(N) by using union find
        space O(N)
        and there is a map solution
     */
    public static int longestConsecutive(int[] nums) {
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        int[] root = new int[n + 1];
        int[] ids = new int[n + 1];

        for (int i = 0; i <= n; i++) {
            root[i] = i;
            ids[i] = 1;
        }
        Map<Integer, Integer> s = new HashMap<>();
        for (int i = 0; i < n; i++) {
            if (s.containsKey(nums[i])) {
                continue;
            }
            s.put(nums[i], i);
        }
        for (Map.Entry<Integer, Integer> e : s.entrySet()) {

            if (s.containsKey(e.getKey() + 1)) {
                union(e.getValue(), s.get(e.getKey() + 1), root, ids);
            }
            if (s.containsKey(e.getKey() - 1)) {
                union(e.getValue(), s.get(e.getKey() - 1), root, ids);
            }
        }
        int max = Integer.MIN_VALUE;
        for (int id : ids) {
            max = Math.max(max, id);
        }
        return max;
    }

    /*
        using map
        each time start building sequence by curnum -1 do not exist.
        then check num + 1 and count the length.
        time O(N)
        space O(N)
     */
    public int longestConsecutiveSet(int[] nums) {
        Set<Integer> num_set = new HashSet<Integer>();
        for (int num : nums) {
            num_set.add(num);
        }

        int longestStreak = 0;

        for (int num : num_set) {
            if (!num_set.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;

                while (num_set.contains(currentNum + 1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }

                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;
    }


    //理论上，从一个完全不相连通的N个对象的集合开始，任意顺序的M次union()调用所需的时间为O(N+Mlg*N)。
    //其中lg*N称为迭代对数（iterated logarithm）。实数的迭代对数是指须对实数连续进行几次对数运算后，
    // 其结果才会小于等于1。这个函数增加非常缓慢，可以视为近似常数（例如2^65535的迭代对数为5）。
    // aveg inverse Ackermann function nearly O(1)
    /*
    Definition: A function of two parameters whose value grows very, very slowly.
     Formal Definition: α(m,n) = min{i≥ 1: A(i, ⌊ m/n⌋) > log2 n} where A(i,j)
     is Ackermann's function.
     */
    private static void union(int a, int b, int[] root, int[] ids) {
        int aRoot = find(a, root);
        int bRoot = find(b, root);
        if (aRoot == bRoot) {
            return;
        }

        //link it to the root with more connections, which will shorten the tree's height
        if (ids[aRoot] < ids[bRoot]) {
            root[aRoot] = root[bRoot];
            ids[bRoot] += ids[aRoot];
        } else {
            root[bRoot] = root[aRoot];
            ids[aRoot] += ids[bRoot];
        }
    }

    //avg  nearly to O(1)
    private static int find(int a, int[] root) {
        while (a != root[a]) {
            //compress the path
            root[a] = root[root[a]];
            a = root[a];
        }
        return a;
    }

    private int countSets(int[] root) {
        int cnt = 0;
        for (int i = 0; i <= root.length - 1; i++) {
            if (find(i, root) == i) {
                cnt++;
            }
        }
        return cnt;
    }

    /*
    lc 556 Next Greater Element III
    similar to next permutation
    Time O(N)
    Space O(N) N for array
    1. find from end, find the first decreasing element
    2. swap the first decreasing element with the number just larger than it from the end
    3.swap this two,
    4. reverse the numbers after the first decreasing number.
     */
    public int nextGreaterElement(int n) {
        char[] a = ("" + n).toCharArray();
        int i = a.length - 2;
        while (i >= 0 && a[i + 1] <= a[i]) {
            i--;
        }
        if (i < 0) {
            return -1;
        }
        int j = a.length - 1;
        while (j >= 0 && a[j] <= a[i]) {
            j--;
        }
        swap(a, i, j);
        reverse(a, i + 1);
        try {
            return Integer.parseInt(new String(a));
        } catch (Exception e) {
            return -1;
        }
    }

    private void reverse(char[] a, int start) {
        int i = start, j = a.length - 1;
        while (i < j) {
            swap(a, i, j);
            i++;
            j--;
        }
    }

    private void swap(char[] a, int i, int j) {
        char temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }

    /*
    503. Next Greater Element II
    Time O(N)
    Space O(N)
    单调栈
     */
    public static int[] nextGreaterElements2(int[] nums) {
        //store index in stack
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        // two pass
        for (int i = 0; i < 2 * nums.length; i++) {
            while (!stack.empty() && nums[i % nums.length] > nums[stack.peek()]) {
                res[stack.pop()] = nums[i % nums.length];
            }
            if (res[i % nums.length] == -1) {
                stack.push(i % nums.length);
            }
        }

        return res;
    }

    //O(n^2)
    public int[] nextGreaterElements2Old(int[] nums) {
        int[] result = new int[nums.length];
        for (int i = 0; i <= nums.length - 1; i++) {
            int j = i;
            int flag = 0;
            while (true) {
                if (flag == 1 && i == j) {
                    result[i] = -1;
                    break;
                }
                j++;
                if (j == nums.length) {
                    j = 0;
                    flag++;
                }
                if (nums[i] < nums[j]) {
                    result[i] = nums[j];
                    break;
                }
            }
        }
        return result;
    }

    /*
    lc 496
    Next Greater Element I
    Time O(N+M)
    Space O(N+M)
     */
    public int[] nextGreaterElement1(int[] findNums, int[] nums) {
        Stack<Integer> stack = new Stack<>();
        HashMap<Integer, Integer> map = new HashMap<>();
        int[] res = new int[findNums.length];
        for (int i = 0; i < nums.length; i++) {
            while (!stack.empty() && nums[i] > stack.peek()) {
                map.put(stack.pop(), nums[i]);
            }
            stack.push(nums[i]);
        }
        while (!stack.empty()) {
            map.put(stack.pop(), -1);
        }
        for (int i = 0; i < findNums.length; i++) {
            res[i] = map.get(findNums[i]);
        }
        return res;
    }


    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int[][] per = new int[4][2];
        int[] a1 = {0, 1};
        int[] a2 = {1, 0};
        int[] a3 = {0, -1};
        int[] a4 = {-1, 0};
        per[0] = a1;
        per[1] = a2;
        per[2] = a3;
        per[3] = a4;
        int c = 0, r = 0;
        boolean[][] visited = new boolean[n][n];
        int[] nextmove = per[0];
        int t = 0;
        for (int i = 0; i < n * n; i++) {
            res[r][c] = i + 1;
            visited[r][c] = true;
            if (!(r + nextmove[0] >= 0 && r + nextmove[0] < n && c + nextmove[1] >= 0
                && c + nextmove[1] < n && !visited[r + nextmove[0]][c + nextmove[1]])) {
                t++;
                nextmove = per[t % 4];
            }
            r += nextmove[0];
            c += nextmove[1];
        }
        return res;
    }

    public String getLastSubString(String s) {
        if (s.length() == 1) {
            return s;
        }

        int currentMaxIndex = 0; //记录最大字符下标
        boolean needCompare = false; //是否有挑战者
        int newMaxIndex = 0; //挑战者下标
        for (int i = 1; i < s.length(); i++) { // 一次循环
            // 如何遍历的字符比当前字符还大，则最大字符下标变为当前下标
            if (s.charAt(i) > s.charAt(currentMaxIndex)) {
                currentMaxIndex = i;
                continue;
            }
            // 如果第一次出现挑战者，记录挑战者 （）
            if (s.charAt(i) == s.charAt(currentMaxIndex) && !needCompare) {
                newMaxIndex = i;
                needCompare = true;
                continue;
            }
            // 有挑战者时，如果挑战成功，则变更擂主
            if (needCompare) {
                if (s.charAt(i) > s.charAt(currentMaxIndex + i - newMaxIndex)) {
                    currentMaxIndex = newMaxIndex;
                    needCompare = false;
                }
            }
        }
        return s.substring(currentMaxIndex);
    }


    public static List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length;
        int n;
        List<Integer> res = new ArrayList<>();
        if (m > 0) {
            n = matrix[0].length;
        } else {
            return res;
        }
        int[][] per = new int[4][2];
        int[] a1 = {0, 1};
        int[] a2 = {1, 0};
        int[] a3 = {0, -1};
        int[] a4 = {-1, 0};
        per[0] = a1;
        per[1] = a2;
        per[2] = a3;
        per[3] = a4;
        int c = 0;
        int r = 0;
        int[] nextmove = per[0];
        boolean[][] visited = new boolean[m][n];
        int t = 0;
        for (int i = 0; i < m * n; i++) {
            res.add(matrix[r][c]);
            visited[r][c] = true;
            int nextr = r + nextmove[0];
            int nextc = c + nextmove[1];
            if (nextc >= n || nextr >= m || nextc < 0 || nextr < 0) {
                t++;
                nextmove = per[t % 4];
            } else if (visited[nextr][nextc]) {
                t++;
                nextmove = per[t % 4];
            }
            r += nextmove[0];
            c += nextmove[1];

        }
        return res;


    }

    private int[] pre, low;
    private int time;

    public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        pre = new int[n];
        low = new int[n];
        time = 0;
        Arrays.fill(pre, -1);
        List<Integer>[] adj = new List[n];
        for (int i = 0; i < n; i++) {
            adj[i] = new ArrayList<>();
        }
        for (List<Integer> connection : connections) {
            adj[connection.get(0)].add(connection.get(1));
            adj[connection.get(1)].add(connection.get(0));
        }
        List<List<Integer>> res = new ArrayList<>();
        dfs(adj, 0, res, -1);
        return res;
    }

    private void dfs(List<Integer>[] adj, int u, List<List<Integer>> res, int parent) {
        pre[u] = low[u] = ++time;
        for (int v : adj[u]) {
            if (v == parent) {
                continue;
            }
            if (pre[v] == -1) {
                dfs(adj, v, res, u);
                low[u] = Math.min(low[u], low[v]);
                if (low[v] > pre[u]) {
                    res.add(Arrays.asList(u, v));
                }
            } else {
                low[u] = Math.min(low[u], pre[v]);
            }
        }
    }

    public static String mostCommonWord(String paragraph, String[] banned) {
        Set<String> banset = new HashSet();
        for (String word : banned) {
            banset.add(word);
        }
        Map<String, Integer> count = new HashMap();

        String ans = "";
        int ansfreq = 0;

        StringBuilder word = new StringBuilder();
        for (char c : paragraph.toCharArray()) {
            if (Character.isLetter(c)) {
                word.append(Character.toLowerCase(c));
            } else if (word.length() > 0) {
                String finalword = word.toString();
                if (!banset.contains(finalword)) {
                    count.put(finalword, count.getOrDefault(finalword, 0) + 1);
                    if (count.get(finalword) > ansfreq) {
                        ans = finalword;
                        ansfreq = count.get(finalword);
                    }
                }
                word = new StringBuilder();
            }
        }

        return ans.equals("") ? word.toString() : ans;
    }

    /*
        283. Move Zeroes
        Time N
        Space 1
     */
    public void moveZeroes(int[] nums) {
        int cur = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[cur] = nums[i];
                cur++;
            }
        }
        while (cur < nums.length) {
            nums[cur] = 0;
            cur++;
        }
    }


    public static String addBinary(String a, String b) {
        int aL = a.length() - 1;
        int bL = b.length() - 1;
        StringBuilder res = new StringBuilder();
        boolean jin = false;
        while (aL >= 0 && bL >= 0) {
            //0+0=96  0+1=97 1+1=98
            switch (a.charAt(aL) + b.charAt(bL)) {
                case 96:
                    res.insert(0, jin ? '1' : '0');
                    jin = false;
                    break;
                case 97:
                    res.insert(0, jin ? '0' : '1');
                    break;
                case 98:
                    res.insert(0, jin ? '1' : '0');
                    jin = true;
                    break;
            }
            aL--;
            bL--;
        }
        while (aL >= 0) {
            if (jin) {
                if (a.charAt(aL) == '1') {
                    res.insert(0, '0');
                } else {
                    res.insert(0, '1');
                    jin = false;
                }
            } else {
                res.insert(0, a.charAt(aL));
            }
            aL--;
        }
        while (bL >= 0) {
            if (jin) {
                if (b.charAt(bL) == '1') {
                    res.insert(0, '0');
                } else {
                    res.insert(0, '1');
                    jin = false;
                }
            } else {
                res.insert(0, b.charAt(bL));
            }
            bL--;
        }
        if (jin) {
            res.insert(0, '1');
        }
        return res.toString();
    }


    int res = 0;
    int[] offRow = new int[]{0, 1, 0, -1};
    int[] offCol = new int[]{1, 0, -1, 0};

    //必须每个空白都踩 0为空白 -1为障碍
    public int uniquePathsIII(int[][] grid) {
        int row = grid.length;
        if (row == 0) {
            return 0;
        }
        int count = 0;
        Pair<Integer, Integer> start = new Pair<>(0, 0);
        int col = grid[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    start = new Pair<>(i, j);
                }
                if (grid[i][j] == 0) {
                    count++;
                }
            }
        }
        backtrackingUniquePath(start.getKey(), start.getValue(), count + 1, grid);
        return res;

    }

    private void backtrackingUniquePath(int row, int col, int count, int[][] grid) {
        int cur = grid[row][col];
        if (cur == 2) {
            if (count == 0) {
                res++;
            }
            return;
        }

        grid[row][col] = -1;
        for (int i = 0; i < 4; i++) {
            int newRow = row + offRow[i];
            int newCol = col + offCol[i];
            if (inGrid(newRow, newCol, grid) && grid[newRow][newCol] != -1) {
                backtrackingUniquePath(newRow, newCol, count - 1, grid);
            }
        }
        grid[row][col] = cur;
    }

    private boolean inGrid(int row, int col, int[][] grid) {
        return row >= 0 && col >= 0 && row < grid.length && col < grid[0].length;
    }


    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int n = obstacleGrid.length;
        int m = obstacleGrid[0].length;
        if (n == 0) {
            return 0;
        }
        int[][] dp = new int[n + 1][m + 1];
        dp[1][1] = obstacleGrid[0][0] == 0 ? 1 : 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                if (obstacleGrid[i - 1][j - 1] == 1) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[n][m];
    }

    public static int uniquePaths(int m, int n) {
        int[][] dp = new int[n + 1][m + 1];
        dp[1][1] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[n][m];
    }

    /*
    lc 39
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        //Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        findResult(candidates, target, new LinkedList<>(), res, 0);
        return res;
    }

    void findResult(int[] candidates, int target, Deque<Integer> nums,
        List<List<Integer>> res, int start) {
        if (target < 0) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList(nums));
        } else {
            for (int i = start; i < candidates.length && target - candidates[i] >= 0; i++) {
                //假如一个数字只用一次
//                if(i>start&&candidates[i]==candidates[i-1])
//                    continue;
                nums.add(candidates[i]);
                findResult(candidates, target - candidates[i], nums, res, i);//只用一次的话为i+1
                nums.removeLast();
            }
        }
    }

    /*
        lc377
        combination sum的总数 数字可以重复，不同次序有效
        1、状态

对于“状态”，我们首先思考能不能就用问题当中问的方式定义状态，上面递归树都画出来了。当然就用问题问的方式。

dp[i] ：对于给定的由正整数组成且不存在重复数字的数组，和为 i 的组合的个数。

思考输出什么？因为状态就是问题当中问的方式而定义的，因此输出就是最后一个状态 dp[n]。

2、状态转移方程

由上面的树形图，可以很容易地写出状态转移方程：

dp[i] = sum{dp[i - num] for num in nums and if i >= num}
注意：在 00 这一点，我们定义 dp[0] = 1 的，它表示如果 nums 里有一个数恰好等于 target，它单独成为 11 种可能。

     */
    public int combinationSum4(int[] nums, int target) {

        int len = nums.length;
        if (len == 0) {
            return 0;
        }
        int[] dp = new int[target + 1];

        // 注意：理解这一句代码的含义
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < len; j++) {
                if (i - nums[j] >= 0) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }
        return dp[target];

    }

    public static int searchInsert(int[] nums, int target) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0] >= target ? 0 : 1;
        }
        int low = 0;
        int high = nums.length - 1;
        while (low <= high) {
            int mid = (low + high) >>> 1;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return low;
    }

    /*
        45. Jump Game II
        Time N
        1
     */
    public static int jump(int[] nums) {
        int end = 0;
        int maxPosition = 0;
        int steps = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            //找能跳的最远的
            maxPosition = Math.max(maxPosition, nums[i] + i);
            if (i == end) { //遇到边界，就更新边界，并且步数加一
                end = maxPosition;
                steps++;
            }
        }
        return steps;
    }


    public static int[] searchRange(int[] nums, int target) {
        int[] res = {-1, -1};
        if (nums.length == 0) {
            return res;
        }

        int pos = bSearch(0, nums.length - 1, target, nums);
        if (pos != -1) {
            int i = pos;
            int j = pos;
            while (i > 0 && nums[i - 1] == nums[pos]) {
                i--;
            }
            while (j < nums.length - 1 && nums[j + 1] == nums[pos]) {
                j++;
            }
            res[0] = i;
            res[1] = j;
        }
        return res;
    }

    //二分搜索
    public static int findFirstBSearch(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int start = 0;
        int end = nums.length - 1;
        while (start + 1 < end) {//0 1
            int mid = start + (end - start) / 2;// mid = 1
            if (nums[mid] >= target) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (nums[start] == target) {
            return start;
        } else if (nums[end] == target) {
            return end;
        }
        return -1;
    }

    public static int bSearch(int low, int high, int target, int[] nums) {
        while (low <= high) {
            int mid = (low + high) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    }


    //找出所有重复的元素
    public static List<Integer> findDuplicates(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for (int num : nums) {
            num = Math.abs(num);
            if (nums[num - 1] > 0) {
                nums[num - 1] = -nums[num - 1];
            } else {
                res.add(num);
            }
        }
        return res;
    }

    //最多只能出现两次
    public static int removeDuplicates2(int[] nums) {
        int length = nums.length;
        if (length <= 2) {
            return length;
        }
        int i = 2;
        int j = 2;
        for (; i < length && j < length; i++) {
            while (j < length && nums[j] == nums[i - 1] && nums[j] == nums[i - 2]) {
                if (j == length - 1) {
                    return i;
                }
                j++;
            }
            nums[i] = nums[j];
            j++;
        }
        return i;
    }

    //最多只能出现一次
    public static int removeDuplicates(int[] nums) {
        if (nums.length <= 1) {
            return nums.length;
        }
        int i = 0;
        int j = 1;
        for (; j <= nums.length - 1 && i < nums.length - 1; i++) {
            if (nums[i] == nums[j]) {
                if (i + 1 == nums.length - 1 || j == nums.length - 1) {
                    return i + 1;
                }
                nums[i + 1] = nums[++j];
                i--;
            } else {
                nums[i + 1] = nums[j];
                j++;
            }
        }
        return i + 1;
    }

    //行和列分别有序 时间复杂度(n+m)
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0) {
            return false;
        }
        if (matrix[0].length == 0) {
            return false;
        }
        int i = 0;
        int j = matrix[0].length - 1;
        while (i < matrix.length && j >= 0) {
            if (target == matrix[i][j]) {
                return true;
            }
            if (target < matrix[i][j]) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }

    /**
     * 126 word ladder list 1.找到每个单词的转移分支，通过*来遮盖某一位来完成这个map 2。双向bfs，每个bfs节点保存当前的所有路径
     * 3.相遇后将当前单词列表加入答案，并完成本次队列里的遍历后结束 4 该function有bug 勿用 //
     */
//    public static List<List<String>> findLadders(String beginWord, String endWord,
//        List<String> wordList) {
//        Set<List<String>> res = new HashSet<>();
//        Map<String, List<String>> comboDic = new HashMap<>();
//        if (!wordList.contains(endWord)) {
//            return new ArrayList<>(res);
//        }
//        if (beginWord.equals(endWord)) {
//            List<String> singleResult = new ArrayList<>();
//            singleResult.add(beginWord);
//            res.add(singleResult);
//            return new ArrayList<>(res);
//        }
//        for (String word : wordList) {
//            for (int i = 0; i < beginWord.length(); i++) {
//                StringBuilder mask = new StringBuilder();
//                mask.append(word.substring(0, i));
//                mask.append("*");
//                mask.append(word.substring(i + 1));
//                List<String> wordListForAMask = comboDic
//                    .getOrDefault(mask.toString(), new ArrayList<>());
//                wordListForAMask.add(word);
//                comboDic.put(mask.toString(), wordListForAMask);
//            }
//        }
//        Queue<Pair<String, List<String>>> qF = new LinkedList<>();
//        Queue<Pair<String, List<String>>> qE = new LinkedList<>();
//        Map<String, List<List<String>>> isVistedF = new HashMap<>();
//        Map<String, List<List<String>>> isVistedE = new HashMap<>();
//        int resSize = Integer.MAX_VALUE;
//        qF.offer(new Pair<>(beginWord, new ArrayList<>(Arrays.asList(new String[]{beginWord}))));
//        qE.offer(new Pair<>(endWord, new ArrayList<>(Arrays.asList(new String[]{endWord}))));
//        List<List<String>> eVisited = new ArrayList<>();
//        List<List<String>> fVisited = new ArrayList<>();
//        eVisited.add(qE.peek().getValue());
//        isVistedE.put(endWord, eVisited);
//        fVisited.add(qF.peek().getValue());
//        isVistedF.put(beginWord, fVisited);
//        while (!qF.isEmpty() && !qE.isEmpty()) {
//            if (!qF.isEmpty()) {
//                Pair<String, List<String>> curPair = qF.remove();
//                if (!(!res.isEmpty() && resSize <= curPair.getValue().size())) {
//                    String curWord = curPair.getKey();
//                    for (int i = 0; i < curPair.getKey().length(); i++) {
//                        StringBuilder mask = new StringBuilder();
//                        mask.append(curWord.substring(0, i));
//                        mask.append("*");
//                        mask.append(curWord.substring(i + 1));
//                        List<String> wordListForAMask = comboDic
//                            .getOrDefault(mask.toString(), new ArrayList<>());
//                        for (String nextWord : wordListForAMask) {
//                            if (curWord.equals((nextWord)) || curPair.getValue()
//                                .contains(nextWord)) {
//                                continue;
//                            }
//                            if (nextWord.equals(endWord)) {
//                                curPair.getValue().add(endWord);
//                                List<String> newRes = new ArrayList<>(
//                                    Arrays.asList(new String[curPair.getValue().size()]));
//                                Collections.copy(newRes, curPair.getValue());
//                                if (newRes.size() <= resSize) {
//                                    res.add(newRes);
//                                    resSize = newRes.size();
//                                }
//                            }
//                            if (isVistedE.containsKey(nextWord)) {
//                                List<String> newRes = new ArrayList<>();
//                                newRes.addAll(curPair.getValue());
//                                List<List<String>> anotherParts = isVistedE.get(nextWord);
//                                for (List<String> anotherPart : anotherParts) {
//                                    List<String> wholeRes = new ArrayList<>(newRes);
//                                    wholeRes.addAll(anotherPart);
//                                    if (wholeRes.size() <= resSize) {
//                                        res.add(wholeRes);
//                                        resSize = wholeRes.size();
//                                    }
//                                }
//                            } else {
//                                List<String> next = new ArrayList<>();
//                                next.addAll(curPair.getValue());
//                                next.add(nextWord);
//                                List<List<String>> allPath = isVistedF
//                                    .getOrDefault(nextWord, new ArrayList<>());
//                                allPath.add(next);
//                                isVistedF.put(nextWord, allPath);
//                                qF.offer(new Pair<>(nextWord, next));
//                            }
//                        }
//                    }
//                }
//            }
//            if (!qE.isEmpty()) {
//                Pair<String, List<String>> curPair = qE.remove();
//                if (!(!res.isEmpty() && resSize <= curPair.getValue().size())) {
//                    String curWord = curPair.getKey();
//                    for (int i = 0; i < curPair.getKey().length(); i++) {
//                        StringBuilder mask = new StringBuilder();
//                        mask.append(curWord.substring(0, i));
//                        mask.append("*");
//                        mask.append(curWord.substring(i + 1));
//                        List<String> wordListForAMask = comboDic
//                            .getOrDefault(mask.toString(), new ArrayList<>());
//                        for (String nextWord : wordListForAMask) {
//                            if (curWord.equals(nextWord) || curPair.getValue().contains(nextWord)) {
//                                continue;
//                            }
//                            if (nextWord.equals(endWord)) {
//                                curPair.getValue().add(endWord);
//                                List<String> newRes = new ArrayList<>(
//                                    Arrays.asList(new String[curPair.getValue().size()]));
//                                Collections.copy(newRes, curPair.getValue());
//                                if (newRes.size() <= resSize) {
//                                    res.add(newRes);
//                                    resSize = newRes.size();
//                                }
//                            }
//                            if (isVistedF.containsKey(nextWord)) {
//                                List<String> newRes = new ArrayList<>(curPair.getValue());
//                                List<List<String>> anotherParts = isVistedF.get(nextWord);
//                                for (List<String> anotherPart : anotherParts) {
//                                    List<String> wholeRes = new ArrayList<>(anotherPart);
//                                    wholeRes.addAll(newRes);
//                                    if (wholeRes.size() <= resSize) {
//                                        res.add(wholeRes);
//                                        resSize = wholeRes.size();
//                                    }
//                                }
//                            } else {
//                                List<String> next = new LinkedList<>();
//                                next.addAll(curPair.getValue());
//                                next.add(0, nextWord);
//                                List<List<String>> allPath = isVistedF
//                                    .getOrDefault(nextWord, new ArrayList<>());
//                                allPath.add(next);
//                                isVistedE.put(nextWord, allPath);
//                                qE.offer(new Pair<>(nextWord, next));
//                            }
//                        }
//                    }
//                }
//
//            }
//        }
//        return new ArrayList<>(res);
//    }
    public List<List<String>> findLadders(String beginWord, String endWord,
        List<String> wordList) {
        List<List<String>> ans = new ArrayList<>();
        if (!wordList.contains(endWord)) {
            return ans;
        }
        //做词典
        Map<String, List<String>> comboDic = new HashMap<>();
        for (String word : wordList) {
            for (int i = 0; i < beginWord.length(); i++) {
                StringBuilder mask = new StringBuilder();
                mask.append(word.substring(0, i));
                mask.append("*");
                mask.append(word.substring(i + 1));
                List<String> wordListForAMask = comboDic
                    .getOrDefault(mask.toString(), new ArrayList<>());
                wordListForAMask.add(word);
                comboDic.put(mask.toString(), wordListForAMask);
            }
        }
        Map<String, Set<String>> wordDict = new HashMap<>();
        wordList.add(beginWord);
        for (String word : wordList) {
            Set<String> set = wordDict.getOrDefault(word, new HashSet<>());
            for (int i = 0; i < beginWord.length(); i++) {
                StringBuilder mask = new StringBuilder();
                mask.append(word.substring(0, i));
                mask.append("*");
                mask.append(word.substring(i + 1));
                List<String> wordListForAMask = comboDic
                    .getOrDefault(mask.toString(), new ArrayList<>());
                set.addAll(wordListForAMask);
            }
            set.remove(word);
            wordDict.put(word, set);
        }
        // 利用 BFS 得到所有的邻居节点
        HashMap<String, ArrayList<String>> map = new HashMap<>();
        bfs(beginWord, endWord, wordList, map, wordDict);
        ArrayList<String> temp = new ArrayList<String>();
        // temp 用来保存当前的路径
        temp.add(beginWord);
        findLaddersHelper(beginWord, endWord, map, temp, ans);
        return ans;
    }

    //dfs find all possible ans
    //此时map里保存了所有可能的最短路径解，只需dfs一遍全找出来
    private void findLaddersHelper(String beginWord, String endWord,
        HashMap<String, ArrayList<String>> map,
        ArrayList<String> temp, List<List<String>> ans) {
        if (beginWord.equals(endWord)) {
            ans.add(new ArrayList<String>(temp));
            return;
        }
        // 得到所有的下一个的节点
        ArrayList<String> neighbors = map.getOrDefault(beginWord, new ArrayList<String>());
        for (String neighbor : neighbors) {
            temp.add(neighbor);
            findLaddersHelper(neighbor, endWord, map, temp, ans);
            temp.remove(temp.size() - 1);
        }
    }

    /*
    利用递归实现了双向搜索
    1.词语列表中删除当前已存的节点（避免走回头路，强制往中间靠拢）
    2.永远从当前层级少的方向搜索（比如从上到下2，从下到上3 则从上到下搜，因为bfs面越窄速度越快）
    3.如果相遇，则停止该条路径，将结果保存在map里，返回。
     */
    private void bfs(String beginWord, String endWord, List<String> wordList,
        HashMap<String, ArrayList<String>> map, Map<String, Set<String>> wordDict) {
        Set<String> set1 = new HashSet<String>();
        set1.add(beginWord);
        Set<String> set2 = new HashSet<String>();
        set2.add(endWord);
        Set<String> wordSet = new HashSet<String>(wordList);
        bfsHelper(set1, set2, wordSet, true, map, wordDict);
    }

    // direction 为 true 代表向下扩展，false 代表向上扩展
    private void bfsHelper(Set<String> set1, Set<String> set2, Set<String> wordSet,
        boolean direction,
        HashMap<String, ArrayList<String>> map, Map<String, Set<String>> wordDict) {
        //set1 为空了，就直接结束,没有能继续搜索的点了
        //比如下边的例子就会造成 set1 为空
    /*	"hot"
		"dog"
		["hot","dog"]*/
        if (set1.isEmpty()) {
            return;
        }
        // 将已经访问过单词删除
        wordSet.removeAll(set1);
        wordSet.removeAll(set2);

        boolean done = false;

        // 保存新扩展得到的节点
        Set<String> set = new HashSet<String>();

        for (String str : set1) {
            for (String word : wordDict.get(str)) {
//遍历每一位来获取同义词
//            for (int i = 0; i < str.length(); i++) {
//                char[] chars = str.toCharArray();
//
//                // 尝试所有字母
//                for (char ch = 'a'; ch <= 'z'; ch++) {
//                    if (chars[i] == ch) {
//                        continue;
//                    }
//                    chars[i] = ch;
//
//                    String word = new String(chars);

                // 根据方向得到 map 的 key 和 val

//                }
//            }
                String key = direction ? str : word;
                String val = direction ? word : str;

                ArrayList<String> list =
                    map.containsKey(key) ? map.get(key) : new ArrayList<>();

                //如果相遇了就保存结果
                if (set2.contains(word)) {
                    done = true;
                    list.add(val);
                    map.put(key, list);
                }

                //如果还没有相遇，并且新的单词在 word 中，那么就加到 map 中
                //在该层已经done之后，不再继续延长path长度，保证搜索结果为最短解
                if (!done && wordSet.contains(word)) {
                    set.add(word);
                    list.add(val);
                    map.put(key, list);
                }
            }


        }

        if (!done) {
            //下次从小的待扩展的方向的set开始扩展，bfs越窄效率越高
            int size2 = set2.size();
            int size = set.size();
            if (size2 > size) {
                bfsHelper(set, set2, wordSet, direction, map, wordDict);
            } else {
                bfsHelper(set2, set, wordSet, !direction, map, wordDict);
            }
        }
    }


    //127. Word Ladder
    //Time O(M*N)  where MM is the length of words and NN is the total number of words in the input word list
    //Space Complexity: O(M * N), to store all MM transformations for each of the NN words,
    // in the all_combo_dict dictionary, same as one directional. But bidirectional reduces the search space.
    // It narrows down because of meeting in the middle.
    public static int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Map<String, List<String>> comboDic = new HashMap<>();
        if (!wordList.contains(endWord)) {
            return 0;
        }
        if (beginWord.equals(endWord)) {
            return 1;
        }
        for (String word : wordList) {
            for (int i = 0; i < beginWord.length(); i++) {
                char[] wordArray = word.toCharArray();
                wordArray[i] = '*';
                String mask = String.valueOf(wordArray);
                List<String> wordListForAMask = comboDic
                    .getOrDefault(String.valueOf(wordArray), new ArrayList<>());
                wordListForAMask.add(word);
                comboDic.put(String.valueOf(wordArray), wordListForAMask);
            }
        }
        Queue<Pair<String, Integer>> qF = new LinkedList<>();
        Queue<Pair<String, Integer>> qE = new LinkedList<>();
        Map<String, Integer> isVistedF = new HashMap<>();
        Map<String, Integer> isVistedE = new HashMap<>();
        qF.offer(new Pair<>(beginWord, 1));
        qE.offer(new Pair<>(endWord, 1));
        isVistedE.put(endWord, 1);
        isVistedF.put(beginWord, 1);
        while (!qF.isEmpty() && !qE.isEmpty()) {
            if (!qF.isEmpty()) {
                Pair<String, Integer> curPair = qF.remove();
                String curWord = curPair.getKey();
                for (int i = 0; i < curPair.getKey().length(); i++) {
                    StringBuilder mask = new StringBuilder();
                    char[] wordArray = curWord.toCharArray();
                    wordArray[i] = '*';
                    mask.append(wordArray);
                    List<String> wordListForAMask = comboDic
                        .getOrDefault(mask.toString(), new ArrayList<>());
                    for (String nextWord : wordListForAMask) {
                        if (isVistedF.containsKey(nextWord)) {
                            continue;
                        }
                        if (nextWord.equals(endWord)) {
                            return curPair.getValue() + 1;
                        }
                        if (isVistedE.containsKey(nextWord)) {
                            return curPair.getValue() + isVistedE.get(nextWord);
                        } else {
                            isVistedF.put(nextWord, curPair.getValue() + 1);
                            qF.offer(new Pair<>(nextWord, curPair.getValue() + 1));
                        }
                    }
                }
            }
            if (!qE.isEmpty()) {
                Pair<String, Integer> curPair = qE.remove();
                String curWord = curPair.getKey();
                for (int i = 0; i < curPair.getKey().length(); i++) {
                    StringBuilder mask = new StringBuilder();
                    mask.append(curWord.substring(0, i));
                    mask.append("*");
                    mask.append(curWord.substring(i + 1));
                    List<String> wordListForAMask = comboDic
                        .getOrDefault(mask.toString(), new ArrayList<>());
                    for (String nextWord : wordListForAMask) {
                        if (isVistedE.containsKey(nextWord)) {
                            continue;
                        }
                        if (nextWord.equals(endWord)) {
                            return curPair.getValue() + 1;
                        }
                        if (isVistedF.containsKey(nextWord)) {
                            return curPair.getValue() + isVistedF.get(nextWord);
                        } else {
                            isVistedE.put(nextWord, curPair.getValue() + 1);
                            qE.offer(new Pair<>(nextWord, curPair.getValue() + 1));
                        }
                    }
                }
            }
        }
        return 0;
    }

    public static int findKthLargest(int[] nums, int k) {

        PriorityQueue<Integer> pq = new PriorityQueue<>(k, Comparator.comparingInt(o -> o));
        for (int num : nums) {
            if (pq.size() == k) {
                if (num > pq.peek()) {
                    pq.poll();
                    pq.add(num);
                }
            } else {
                pq.add(num);
            }
        }
        return pq.peek();
    }

    public static int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        int p = 1;
        for (int i = 0; i < nums.length; i++) {
            res[i] = p;
            p = p * nums[i];
        }
        p = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            res[i] = res[i] * p;
            p = p * nums[i];
        }
        return res;
    }

    /*
    lc 31. Next Permutation
    Time O(N)
    space O(1)
    1. find from end, find the first decreasing element
    2. swap the first decreasing element with the number just larger than it from the end
    3.swap this two,
    4. reverse the numbers after the first decreasing number.
     */
    public static void nextPermutation(int[] nums) {
        if (nums.length <= 1) {
            return;
        }
        int i = nums.length - 1;
        while (nums[i] <= nums[i - 1]) {
            if (i == 1) {
                Arrays.sort(nums);
                return;
            }
            i--;
        }
        int j = i + 1;
        while (j < nums.length && nums[i - 1] < nums[j]) {
            j++;
        }
        swap(nums, i - 1, j - 1);
        reverse(nums, i);
    }

    private static void reverse(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    private static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    /*
        University Career Fair
     */
    public static int maxEvents(List<Integer> arrival, List<Integer> duration) {
        List<Pair<Integer, Integer>> input = new ArrayList<>();

        for (int i = 0; i < arrival.size(); i++) {
            input.add(new Pair(arrival.get(i), duration.get(i) + arrival.get(i)));
        }

        Collections.sort(input, Comparator.comparingInt(Pair::getValue));

        int max = 0;
        int end = 0;
        for (int i = 0; i < input.size(); i++) {
            if (end <= input.get(i).getKey()) {
                max++;
                end = input.get(i).getValue();
            }
        }
        return max;
    }

    /*
        1283. Find the Smallest Divisor Given a Threshold
        Time N+logK*N
        space 1
     */
    public int smallestDivisor(int[] A, int threshold) {
        int left = 1, right = (int) 1e5;
        for (int num : A) {
            left = Math.min(num, left);
            right = Math.max(num, right);
        }
        while (left < right) {
            int m = (left + right) / 2, sum = 0;
            for (int i : A) {
                sum += (i + m - 1) / m;
            }
            if (sum > threshold) {
                left = m + 1;
            } else {
                right = m;
            }
        }
        return left;
    }

    public static int minMeetingRooms(int[][] intervals) {
        if (intervals.length == 0) {
            return 0;
        }
        PriorityQueue<Integer> meetingQueue = new PriorityQueue<Integer>(intervals.length,
            Comparator.comparingInt(o -> o));
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        meetingQueue.add(intervals[0][1]);
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= meetingQueue.peek()) {
                meetingQueue.poll();
            }
            meetingQueue.add(intervals[i][1]);
        }
        return meetingQueue.size();
    }

    public static boolean canAttendMeetings(int[][] intervals) {
        List<int[]> meetings = new ArrayList<>(Arrays.asList(intervals));
        Collections.sort(meetings, (m1, m2) -> {
            if (m1[0] < m2[0]) {
                return -1;
            } else if (m1[0] > m2[0]) {
                return 1;
            } else {
                if (m1[1] >= m2[1]) {
                    return 1;
                } else {
                    return -1;
                }
            }
        });
        for (int i = 0; i < meetings.size() - 1; i++) {
            int[] m1 = meetings.get(i);
            int[] m2 = meetings.get(i + 1);
            if (m1[1] > m2[0]) {
                return false;
            }
        }
        return true;
    }

    public static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> numsList = new ArrayList<>();
        for (int num : nums) {
            numsList.add(num);
        }
        findPermutation(0, res, numsList);
        return res;
    }

    public static void findPermutation(int first, List<List<Integer>> resList, List<Integer> nums) {
        if (first == nums.size() - 1) {
            resList.add(nums);
        } else {
            for (int i = first; i < nums.size(); i++) {
                List<Integer> nums1 = new ArrayList<>(nums);
                swap(nums1, first, i);
                findPermutation(first + 1, resList, nums1);
            }
        }
    }

    public static void swap(List<Integer> nums, int fir, int i) {
        int temp = nums.get(fir);
        nums.set(fir, nums.get(i));
        nums.set(i, temp);
    }

    //O(N2)
    public static Node copyRandomListOld(Node head) {
        if (head == null) {
            return null;
        }
        List<Node> src = new LinkedList<>();
        Node cur = head;
        while (cur != null) {
            src.add(cur);
            cur = cur.next;
        }
        if (src.size() == 1) {
            Node n = new Node();
            n.val = src.get(0).val;
            if (head.random != null) {
                n.random = n;
            } else {
                n.random = null;
            }
            return n;
        }
        int[] randomIndex = new int[src.size()];
        List<Node> target = new LinkedList<>();
        for (int i = 0; i < randomIndex.length; i++) {
            Node n = new Node();
            n.val = src.get(i).val;
            target.add(n);
            if (src.get(i).random != null) {
                randomIndex[i] = src.indexOf(src.get(i).random);
            } else {
                randomIndex[i] = -1;
            }
        }
        for (int i = 0; i < target.size() - 1; i++) {
            target.get(i).next = target.get(i + 1);
            target.get(i).random = randomIndex[i] != -1 ? target.get(randomIndex[i]) : null;
        }
        target.get(target.size() - 1).random =
            randomIndex[target.size() - 1] != -1 ? target.get(randomIndex[target.size() - 1])
                : null;
        return target.get(0);
    }


    Map<Node, Node> nodeDict = new HashMap<>();

    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }
        if (nodeDict.containsKey(head)) {
            return nodeDict.get(head);
        }
        Node newNode = new Node(head.val);
        nodeDict.put(head, newNode);
        newNode.next = copyRandomList(head.next);
        newNode.random = copyRandomList(head.random);
        return newNode;
    }

    //time & space O(N) using a dict to have old node-> new node
    //interweave 交织法可以O(1) space
    //将新的node变成old node.next 然后新的node的random就是old node.random.next
    //之后再unweave 变成一个正常链表

    public Node copyRandomListInterweave(Node head) {

        if (head == null) {
            return null;
        }

        // Creating a new weaved list of original and copied nodes.
        Node ptr = head;
        while (ptr != null) {

            // Cloned node
            Node newNode = new Node(ptr.val);

            // Inserting the cloned node just next to the original node.
            // If A->B->C is the original linked list,
            // Linked list after weaving cloned nodes would be A->A'->B->B'->C->C'
            newNode.next = ptr.next;
            ptr.next = newNode;
            ptr = newNode.next;
        }

        ptr = head;

        // Now link the random pointers of the new nodes created.
        // Iterate the newly created list and use the original nodes' random pointers,
        // to assign references to random pointers for cloned nodes.
        while (ptr != null) {
            ptr.next.random = (ptr.random != null) ? ptr.random.next : null;
            ptr = ptr.next.next;
        }

        // Unweave the linked list to get back the original linked list and the cloned list.
        // i.e. A->A'->B->B'->C->C' would be broken to A->B->C and A'->B'->C'
        Node ptr_old_list = head; // A->B->C
        Node ptr_new_list = head.next; // A'->B'->C'
        Node head_old = head.next;
        while (ptr_old_list != null) {
            ptr_old_list.next = ptr_old_list.next.next;
            ptr_new_list.next = (ptr_new_list.next != null) ? ptr_new_list.next.next : null;
            ptr_old_list = ptr_old_list.next;
            ptr_new_list = ptr_new_list.next;
        }
        return head_old;
    }

    /*
        79. Word Search
        dfs and backtracking
        Time O(M*4*3^(L-1)) 4 directions,at first move, then for the next L-1 move,it will have 3 directions.
        M is the cell nums which is n*m
        space O(M)
     */
    //search word in 2d board
    public static boolean exist(char[][] board, String word) {
        int row = board.length;
        if (row == 0) {
            return false;
        }
        int col = board[0].length;
        if (row * col < word.length()) {
            return false;
        }
        char[] chars = word.toCharArray();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (chars[0] == board[i][j]) {
                    if (searchAround(i, j, chars, 1, board, null)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    static boolean searchAround(int i, int j, char[] chars, int t, char[][] board,
        boolean[][] beenTo) {
        int row = board.length;
        int col = board[0].length;
        if (beenTo == null) {
            beenTo = new boolean[row][col];
        }
        beenTo[i][j] = true;
        if (t == chars.length) {
            return true;
        }
        if (i < row - 1 && board[i + 1][j] == chars[t] && !beenTo[i + 1][j] && searchAround(i + 1,
            j,
            chars, t + 1, board, beenTo)) {
            return true;
        }
        if (i > 0 && board[i - 1][j] == chars[t] && !beenTo[i - 1][j] && searchAround(i - 1, j,
            chars,
            t + 1, board, beenTo)) {
            return true;
        }
        if (j > 0 && board[i][j - 1] == chars[t] && !beenTo[i][j - 1] && searchAround(i, j - 1,
            chars,
            t + 1, board, beenTo)) {
            return true;
        }
        if (j < col - 1 && board[i][j + 1] == chars[t] && !beenTo[i][j + 1] && searchAround(i,
            j + 1,
            chars, t + 1, board, beenTo)) {
            return true;
        }
        beenTo[i][j] = false;
        return false;
    }

    /*
        49. Group Anagrams
        Time N * klogk  k = avg length of strs
        Space n
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] sArr = s.toCharArray();
            Arrays.sort(sArr);
            String key = new String(sArr);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(s);
            map.put(key, list);
        }
        List<List<String>> res = new ArrayList<>();
        res.addAll(map.values());
        return res;
    }

    //Time NK
    public List<List<String>> groupAnagramsByCharCount(String[] strs) {
        if (strs.length == 0) {
            return new ArrayList();
        }
        Map<String, List> ans = new HashMap<String, List>();
        int[] count = new int[26];
        for (String s : strs) {
            Arrays.fill(count, 0);
            for (char c : s.toCharArray()) {
                count[c - 'a']++;
            }

            StringBuilder sb = new StringBuilder("");
            for (int i = 0; i < 26; i++) {
                sb.append('#');
                sb.append(count[i]);
            }
            String key = sb.toString();
            if (!ans.containsKey(key)) {
                ans.put(key, new ArrayList());
            }
            ans.get(key).add(s);
        }
        return new ArrayList(ans.values());
    }


    //旋转数组中搜索
    public static int searchInRotatedArray(int[] nums, int target) {
        if (nums.length == 0) {
            return -1;
        }
        return findTargetInRotatedArray(0, nums.length - 1, target, nums);
    }

    public static int findTargetInRotatedArray(int start, int end, int target, int[] nums) {
        if (start == end || start + 1 == end) {
            if (nums[start] == target) {
                return start;
            } else if (nums[end] == target) {
                return end;
            } else {
                return -1;
            }
        }
        int mid = (end + start) / 2;
        //mid 小于target
        if (nums[mid] < target) {
            //end 大于等于target 说明mid 到end 是递增序列 在后半段找
            if (nums[end] >= target) {
                return findTargetInRotatedArray(mid, end, target, nums);
            }
            //end 小于target 说明后半段中间有拐头 两边都有可能要搜
            else {
                int res = findTargetInRotatedArray(start, mid, target, nums);
                return res != -1 ? res : findTargetInRotatedArray(mid, end, target, nums);
            }

        }
        //mid比target大
        else if (nums[mid] > target) {
            //start小于target 说明前半段递增，直接咱前半段里找
            if (nums[start] <= target) {
                return findTargetInRotatedArray(start, mid, target, nums);
            }
            //start比target大说明前半段可能有拐点，两边都有可能
            else {
                int res = findTargetInRotatedArray(mid, end, target, nums);
                return res != -1 ? res : findTargetInRotatedArray(start, mid, target, nums);
            }
        } else {
            return mid;
        }
    }

    public int maxProfitWithCD(int[] prices) {

        int n = prices.length;
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        int dp_pre_0 = 0; // 代表 dp[i-2][0]
        for (int i = 0; i < n; i++) {
            int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, dp_pre_0 - prices[i]);
            dp_pre_0 = temp;
        }
        return dp_i_0;


    }

    //sell stock 3， only allow do K transaction
    public int maxProfit3(int k, int[] nums) {

        //initial state dim1 - day; dim2 - transacation number
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (k > nums.length) {
            return maxProfit2(nums);
        }
        int[][] dp = new int[nums.length][k + 1];
        // dp[i][0] = 0
        for (int time = 1; time <= k; time++) {
            int max = Integer.MIN_VALUE;
            for (int day = 1; day < nums.length; day++) {

                max = Math.max(dp[day - 1][time - 1] - nums[day - 1], max);
                dp[day][time] = Math.max(dp[day - 1][time], max + nums[day]);
            }
        }
        return dp[nums.length - 1][k];
    }

    //sell stock 1， only allow do one transaction
    public static int maxProfit1(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int buyIn = Integer.MAX_VALUE;
        int sellOut = Integer.MIN_VALUE;
        int maxPro = Integer.MIN_VALUE;
        for (int i : nums) {
            if (i < buyIn) {
                buyIn = i;
                sellOut = Integer.MIN_VALUE;
            }
            sellOut = i > sellOut ? i : sellOut;
            maxPro = maxPro > sellOut - buyIn ? maxPro : sellOut - buyIn;
        }
        return maxPro;
    }

    //sell stock2, do many transaction, but sell before buy
    public int maxProfit2(int[] prices) {
        int profit = 0;
        int minPrice = Integer.MAX_VALUE;
        for (int i : prices) {
            if (i > minPrice) {
                //sell it
                profit += i - minPrice;
            }
            minPrice = i;
        }
        return profit;
    }

    int maxProfit2DP(int[] prices) {
        int n = prices.length;
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, temp - prices[i]);
        }
        return dp_i_0;
    }

    public int maxProfitWithFee(int[] prices, int fee) {

        int n = prices.length;
        int dp_i_0 = 0, dp_i_1 = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int temp = dp_i_0;
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, temp - prices[i] - fee);
        }
        return dp_i_0;
    }


    /*
    lc 56. Merge Intervals
    sort the intervals,then merge the neighboring elements. if merge successfully, don't move the
    Time O(nlogn)
    Space O(N)
     */
    public static int[][] merge(int[][] intervals) {
        List<int[]> s = new ArrayList<>(Arrays.asList(intervals));
        Collections.sort(s, (s1, s2) -> s1[0] - s2[0]);
        for (int i = 0; i < s.size() - 1; i++) {
            int j = i + 1;
            if (canMerge(s.get(i), s.get(j))) {
                int[] newArray = {s.get(i)[0],
                    s.get(j)[1] > s.get(i)[1] ? s.get(j)[1] : s.get(i)[1]};
                s.set(i, newArray);
                s.remove(s.get(j));
                i = i - 1;
            }
        }
        int[][] result = new int[s.size()][2];
        for (int i = 0; i < result.length; i++) {
            result[i] = s.get(i);
        }
        return result;
    }

    private static boolean canMerge(int[] a, int[] b) {
        return (a[0] >= b[0] && a[0] <= b[1]) || (a[1] >= b[0] && a[1] <= b[1]) || (a[0] <= b[0]
            && a[1] >= b[0]) || (a[0] <= b[1] && a[1] >= b[1]);
    }


    public static void swap(int i, int j, int[][] points) {
        int temp[];
        temp = points[i];
        points[i] = points[j];
        points[j] = temp;
    }

    public static int dis(int i, int j) {
        return i * i + j * j;
    }

    /*
        200. Number of Islands
        Time O(N) since each element will be visited once.
        Space O(N)
     */
    int count = 0;

    public int numIslands(char[][] grid) {
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }

    private void dfs(char[][] grid, int curRow, int curCol) {
        grid[curRow][curCol] = 0;
        for (int i = 0; i < 4; i++) {
            int newRow = dir[i][0] + curRow;
            int newCol = dir[i][1] + curCol;
            if (isInGrid(newRow, newCol, grid) && grid[newRow][newCol] == '1') {
                dfs(grid, newRow, newCol);
            }
        }
    }

    private boolean isInGrid(int row, int col, char[][] grid) {
        return row >= 0 && row < grid.length && col >= 0 && col < grid[0].length;
    }


    /*
    lc 21. Merge Two Sorted Lists
    Time O(n+m)
    Space(1)
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode cur = head;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        cur.next = l1 == null ? l2 : l1;
        return head.next;
    }

    public static boolean isValid(String s) {
        Map<Character, Character> data = new HashMap<>();
        data.put('{', '}');
        data.put('[', ']');
        data.put('(', ')');
        Stack<Character> stack = new Stack();
        for (char c : s.toCharArray()) {
            if (stack.size() != 0 && !data.containsKey(stack.peek())) {
                return false;
            }
            if (stack.size() != 0 && c == data.get(stack.peek())) {
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }

    public static List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> result = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            if (i == nums.length - 3) {
                break;
            }
            for (int j = i + 1; j < nums.length; j++) {
                int f = j + 1;
                int l = nums.length - 1;
                int sum = nums[i] + nums[j];
                while (f < l) {
                    if (sum + nums[f] + nums[l] < target) {
                        f++;
                    } else if (sum + nums[f] + nums[l] > target) {
                        l--;
                    } else {
                        List<Integer> singleRes = new LinkedList<>();
                        singleRes.add(nums[i]);
                        singleRes.add(nums[j]);
                        singleRes.add(nums[l]);
                        singleRes.add(nums[f]);
                        result.add(singleRes);
                    }
                }
            }
        }
        return result;
    }

    public static int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int dis = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length - 2; i++) {
            int f = i + 1;
            int l = nums.length - 1;
            while (f < l) {
                if (nums[i] + nums[f] + nums[l] - target < 0) {
                    dis = Math.abs(nums[i] + nums[f] + nums[l] - target) < Math.abs(dis - target) ?
                        nums[i]
                            + nums[f] + nums[l] : dis;
                    f++;
                } else if (nums[i] + nums[f] + nums[l] - target > 0) {
                    dis = Math.abs(nums[i] + nums[f] + nums[l] - target) < Math.abs(dis - target) ?
                        nums[i]
                            + nums[f] + nums[l] : dis;
                    l--;
                } else {
                    return target;
                }
            }
        }
        return dis;
    }

    public static List<List<Integer>> threeSum(int[] nums) {

        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int k = 0; k < nums.length - 2; k++) {
            if (nums[k] > 0) {
                break;
            }
            if (k > 0 && nums[k] == nums[k - 1]) {
                continue;
            }
            int i = k + 1, j = nums.length - 1;
            while (i < j) {
                int sum = nums[k] + nums[i] + nums[j];
                if (sum < 0) {
                    while (i < j && nums[i] == nums[++i]) {
                    }
                } else if (sum > 0) {
                    while (i < j && nums[j] == nums[--j]) {
                    }
                } else {
                    res.add(new ArrayList<Integer>(Arrays.asList(nums[k], nums[i], nums[j])));
                    while (i < j && nums[i] == nums[++i]) {
                    }
                    while (i < j && nums[j] == nums[--j]) {
                    }
                }
            }
        }
        return res;
    }

    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> hm = new HashMap();
        int[] result = new int[2];
        for (int i = 0; i < nums.length; i++) {
            hm.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            if (hm.containsKey(target - nums[i])) {
                if (hm.get(target - nums[i]) == i) {
                    continue;
                }
                result[0] = i;
                result[1] = hm.get(target - nums[i]);
                return result;
            }
        }
        return null;
    }


    static int getLength(ListNode l1) {
        int res = 0;
        while (l1 != null) {
            res++;
            l1 = l1.next;
        }
        return res;
    }

    //数字个位在最后
    public static ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
        Stack<Integer> s1 = new Stack<Integer>();
        Stack<Integer> s2 = new Stack<Integer>();

        while (l1 != null) {
            s1.push(l1.val);
            l1 = l1.next;
        }
        while (l2 != null) {
            s2.push(l2.val);
            l2 = l2.next;
        }

        int sum = 0;
        ListNode list = new ListNode(0);
        while (!s1.empty() || !s2.empty()) {
            if (!s1.empty()) {
                sum += s1.pop();
            }
            if (!s2.empty()) {
                sum += s2.pop();
            }
            list.val = sum % 10;
            ListNode head = new ListNode(sum / 10);
            head.next = list;
            list = head;
            sum /= 10;
        }

        return list.val == 0 ? list.next : list;
    }

    //lc 2 数字个位在链表第一个
    //T&S O(max(m,n))
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {

        ListNode ln = new ListNode((l1.val + l2.val) % 10);
        ListNode first = ln;
        int rem = (l1.val + l2.val) / 10;
        while (l1.next != null || l2.next != null || rem != 0) {
            ln.next = new ListNode(0);
            ln = ln.next;
            if (l1.next != null && l2.next != null) {
                l1 = l1.next;
                l2 = l2.next;
                ln.val = (l1.val + l2.val + rem) % 10;
                rem = (l1.val + l2.val + rem) / 10;
            } else if (l1.next != null) {
                l1 = l1.next;
                ln.val = (l1.val + rem) % 10;
                rem = (l1.val + rem) / 10;
            } else if (l2.next != null) {
                l2 = l2.next;
                ln.val = (l2.val + rem) % 10;
                rem = (l2.val + rem) / 10;
            } else if (rem != 0) {
                ln.val = rem;
                rem = rem / 10;
            }
        }
        return first;
    }


    /*
    lc 3 Longest Substring Without Repeating Characters
    time O(N)
     */
    public int lengthOfLongestSubstring(String s) {
        int i = 0;
        if (s.length() == 0) {
            return 0;
        }
        if (s.length() == 1) {
            return 1;
        }
        Map<Character, Integer> t = new HashMap<>();
        int maxLength = Integer.MIN_VALUE;
        int j = i;
        while (j < s.length()) {
            if (t.containsKey(s.charAt(j))) {
                int index = t.get(s.charAt(j));
                i = Math.max(i, index + 1);
            }
            maxLength = Math.max(maxLength, j - i + 1);
            t.put(s.charAt(j), j);
            j++;

        }
        return maxLength;
    }


    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int medIndex = (nums1.length + nums2.length) / 2;
        int v1 = 0, v2 = 0;
        int i = 0, t1 = 0, t2 = 0;
        while (t1 < nums1.length && t2 < nums2.length) {
            if (i > medIndex) {
                break;
            }
            if (nums1[t1] < nums2[t2]) {
                v1 = v2;
                v2 = nums1[t1];
                t1++;
                i++;
            } else {
                v1 = v2;
                v2 = nums2[t2];
                t2++;
                i++;
            }
        }
        while (t1 < nums1.length) {
            if (i > medIndex) {
                break;
            }
            v1 = v2;
            v2 = nums1[t1];
            t1++;
            i++;
        }
        while (t2 < nums2.length) {
            if (i > medIndex) {
                break;
            }
            v1 = v2;
            v2 = nums2[t2];
            t2++;
            i++;
        }

        if ((nums1.length + nums2.length) % 2 == 0) {
            return (v1 + v2) / 2.0;
        } else {
            return v2;
        }
    }

    public static String reverseWords(String s) {
        s = s.trim();
        String[] words = s.split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            sb.append(words[i]);
            if (i != 0) {
                sb.append(" ");
            }
        }
        return sb.toString();
    }

    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenter(String s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        return R - L - 1;
    }


    //zig zag zigzag String
    public static String convertZ(String s, int numRows) {
        if (numRows == 1 || s.length() < numRows) {
            return s;
        } else {
            int i = 0;
            int t = 0;
            StringBuilder[] results = new StringBuilder[numRows];
            boolean flag = true;
            while (t < s.length()) {
                if (results[i] == null) {
                    results[i] = new StringBuilder();
                }
                results[i].append(s.charAt(t));
                t++;
                if (flag) {
                    i++;
                } else {
                    i--;
                }
                if (i < 0) {
                    i = 1;
                    flag = true;
                } else if (i == numRows) {
                    i = numRows - 2;
                    flag = false;
                }
            }
            String res = "";
            for (StringBuilder sb : results) {
                res += sb.toString();
            }
            return res;
        }

    }

    public static int reverse(int x) {
        boolean flag = true;
        if (x < 0) {
            flag = false;
        }
        x = Math.abs(x);
        List<Integer> L = new ArrayList<>();
        while (x != 0) {
            L.add(x % 10);
            x = x / 10;
        }
        int result = 0;
        for (int i = 0; i < L.size(); i++) {
            result += L.get(i) * Math.pow(10, L.size() - i - 1);
        }

        return result;
    }

    /*
        50. Pow(x, n)
        x的n次方
        time(logn)
        space 1
     */
    private double fastPow(double x, long n) {
        if (n == 0) {
            return 1.0;
        }
        double half = fastPow(x, n / 2);
        if (n % 2 == 0) {
            return half * half;
        } else {
            return half * half * x;
        }
    }

    public double myPow(double x, int n) {
        long N = n;
        if (N < 0) {
            x = 1 / x;
            N = -N;
        }

        return fastPow(x, N);
    }

    //atoi string 转 int
    /*
    lc 8 String to Integer (atoi)
     */
    public static int myAtoi(String str) {
        if (str == null) {
            return 0;
        }
        // 偷懒做法，去掉空格，也可以用while循环来做
        String temp = str.trim();
        if (temp == "" || temp.length() == 0) {
            return 0;
        }
        boolean flag = true;
        char[] bits = temp.toCharArray();
        int i = 0;
        int res = 0;
        int bit = 0;
        if (bits[0] == '-') {
            flag = false;
        }
        if (bits[0] == '+' || bits[0] == '-') {
            i++;
        }
        for (; i < bits.length; i++) {
            if (bits[i] >= '0' && bits[i] <= '9') {
                bit = bits[i] - '0';
            } else {
                break;
            }
            // 这里巧妙的应用了如果溢出就取最大值 Integer.MAX_VALUE 或 Integer.MIN_VALUE
            if (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && bit > 7)) {
                return flag ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            res = res * 10 + bit;
        }
        return flag ? res : -res;
    }

    public static boolean isMatch(String text, String pattern) {
        boolean[][] dp = new boolean[text.length() + 1][pattern.length() + 1];
        dp[text.length()][pattern.length()] = true;

        for (int i = text.length(); i >= 0; i--) {
            for (int j = pattern.length() - 1; j >= 0; j--) {
                boolean first_match = (i < text.length() &&
                    (pattern.charAt(j) == text.charAt(i) ||
                        pattern.charAt(j) == '.'));
                if (j + 1 < pattern.length() && pattern.charAt(j + 1) == '*') {
                    dp[i][j] = dp[i][j + 2] || first_match && dp[i + 1][j];
                } else {
                    dp[i][j] = first_match && dp[i + 1][j + 1];
                }
            }
        }
        return dp[0][0];
    }


    public static String intToRoman(int num) {
        Map<Integer, String> dic = new HashMap<>();
        dic.put(1, "I");
        dic.put(5, "V");
        dic.put(10, "X");
        dic.put(50, "L");
        dic.put(100, "C");
        dic.put(500, "D");
        dic.put(1000, "M");
        StringBuilder result = new StringBuilder();
        int t = 0;
        while (num != 0) {
            int cur = num % 10;
            num = num / 10;
            if (1 <= cur && cur < 4) {
                while (cur != 0) {
                    result.insert(0, dic.get((int) Math.pow(10, t)));
                    cur--;
                }
            } else if (cur == 4 || cur == 9) {
                result.insert(0, dic.get((cur + 1) * (int) Math.pow(10, t)));
                result.insert(0, dic.get((int) Math.pow(10, t)));
            } else if (cur >= 5) {
                result.insert(0, dic.get(5 * (int) Math.pow(10, t)));
                while (cur - 5 != 0) {
                    result.insert(1, dic.get((int) Math.pow(10, t)));
                    cur--;
                }
            }
            t++;
        }
        return result.toString();
    }

    private static String longestCommonPrefix(String[] strs) {
        int cur = 0;
        boolean flag = false;
        if (strs.length == 0) {
            return "";
        }
        while (cur < strs[0].length()) {
            char fir = strs[0].charAt(cur);
            for (String str : strs) {
                if (cur >= str.length() || fir != str.charAt(cur)) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                break;
            }
            cur++;
        }

        if (cur == 0) {
            return "";
        } else {
            return strs[0].substring(0, cur);
        }
    }

    //电话 phone
    private static List<String> letterCombinations(String digits) {
        Map<Character, char[]> tele = new HashMap<>();
        char[] n1 = {};
        char[] n2 = {'a', 'b', 'c'};
        char[] n3 = {'d', 'e', 'f'};
        char[] n4 = {'g', 'h', 'i'};
        char[] n5 = {'j', 'k', 'l'};
        char[] n6 = {'m', 'n', 'o'};
        char[] n7 = {'p', 'q', 'r', 's'};
        char[] n8 = {'t', 'u', 'v'};
        char[] n9 = {'w', 'x', 'y', 'z'};
        tele.put('2', n2);
        tele.put('3', n3);
        tele.put('4', n4);
        tele.put('5', n5);
        tele.put('6', n6);
        tele.put('7', n7);
        tele.put('8', n8);
        tele.put('9', n9);
        List<String> result = new LinkedList<>();
        for (int i = 0; i < digits.length(); i++) {
            if (result.size() == 0) {
                for (char cur : tele.get(digits.charAt(i))) {
                    String newResult = new String(String.valueOf(cur));
                    result.add(newResult);
                }
            } else {
                char[] chars = tele.get(digits.charAt(i));
                List<String> copyList = new LinkedList<>(Arrays.asList(new String[result.size()]));
                Collections.copy(copyList, result);
                for (int n = 0; n < result.size(); n++) {
                    result.set(n, result.get(n) + chars[0]);
                }
                for (int t = 1; t < chars.length; t++) {
                    List<String> newList = new LinkedList<>();
                    for (String rec : copyList) {
                        newList.add(rec + chars[t]);
                    }
                    result.addAll(newList);
                }
            }
        }
        return result;
    }
}

/*
215. Kth Largest Element in an Array
time O(N)
space O(1)
 */
class KthLargestElementInAnArray {

    int[] nums;

    public void swap(int a, int b) {
        int tmp = this.nums[a];
        this.nums[a] = this.nums[b];
        this.nums[b] = tmp;
    }


    public int partition(int left, int right, int pivot_index) {
        int pivot = this.nums[pivot_index];
        // 1. move pivot to end
        swap(pivot_index, right);
        int store_index = left;

        // 2. move all smaller elements to the left
        for (int i = left; i <= right; i++) {
            if (this.nums[i] < pivot) {
                swap(store_index, i);
                store_index++;
            }
        }

        // 3. move pivot to its final place
        swap(store_index, right);

        return store_index;
    }

    public int quickselect(int left, int right, int k_smallest) {
    /*
    Returns the k-th smallest element of list within left..right.
    */

        if (left == right) // If the list contains only one element,
        {
            return this.nums[left];  // return that element
        }

        // select a random pivot_index
        int pivot_index = (left + right) / 2;

        pivot_index = partition(left, right, pivot_index);

        // the pivot is on (N - k)th smallest position
        if (k_smallest == pivot_index) {
            return this.nums[k_smallest];
        }
        // go left side
        else if (k_smallest < pivot_index) {
            return quickselect(left, pivot_index - 1, k_smallest);
        }
        // go right side
        return quickselect(pivot_index + 1, right, k_smallest);
    }

    public int findKthLargest(int[] nums, int k) {
        this.nums = nums;
        int size = nums.length;
        // kth largest is (N - k)th smallest
        return quickselect(0, size - 1, size - k);
    }
}


class ListNode {

    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}

class Node {

    public int val;
    public Node next;
    public Node random;
    public List<Node> neighbors;

    public Node() {
    }

    public Node(int val) {
        this.val = val;
    }

    public Node(int _val, Node _next, Node _random) {
        val = _val;
        next = _next;
        random = _random;
    }
}


class TrieNodeArray {

    TrieNodeArray[] children;
    String word = null;

    public TrieNodeArray() {
        children = new TrieNodeArray[26];
    }
}


class TrieNode {

    Map<Character, TrieNode> children;
    String word = null;

    public TrieNode() {
        children = new HashMap<>();
    }
}

/*
lc 212. Word Search II
dfs and backtracking
Time O(M*4*3^(L-1)) 4 directions,at first move, then for the next L-1 move,it will have 3 directions.
M is the cell nums which is n*m
Space O(letter numbers)
 */
//use trie to search all words in matrix
class WordSearch2 {

    static int[] rowOffset = {-1, 0, 1, 0};
    static int[] colOffset = {0, 1, 0, -1};
    char[][] _board = null;
    ArrayList<String> _result = new ArrayList<String>();

    public List<String> findWords(char[][] board, String[] words) {

        // Step 1). Construct the Trie
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode node = root;

            for (Character letter : word.toCharArray()) {
                if (node.children.containsKey(letter)) {
                    node = node.children.get(letter);
                } else {
                    TrieNode newNode = new TrieNode();
                    node.children.put(letter, newNode);
                    node = newNode;
                }
            }
            node.word = word;  // store words in Trie
        }

        this._board = board;
        // Step 2). Backtracking starting for each cell in the board
        for (int row = 0; row < board.length; ++row) {
            for (int col = 0; col < board[row].length; ++col) {
                if (root.children.containsKey(board[row][col])) {
                    backtracking(row, col, root);
                }
            }
        }

        return this._result;
    }

    private void backtracking(int row, int col, TrieNode parent) {
        Character letter = this._board[row][col];
        TrieNode currNode = parent.children.get(letter);

        // check if there is any match
        if (currNode.word != null) {
            this._result.add(currNode.word);
            currNode.word = null;
        }

        // mark the current letter before the EXPLORATION
        this._board[row][col] = '#';

        // explore neighbor cells in around-clock directions: up, right, down, left

        for (int i = 0; i < 4; ++i) {
            int newRow = row + rowOffset[i];
            int newCol = col + colOffset[i];
            if (newRow < 0 || newRow >= this._board.length || newCol < 0
                || newCol >= this._board[0].length) {
                continue;
            }
            if (currNode.children.containsKey(this._board[newRow][newCol])) {
                backtracking(newRow, newCol, currNode);
            }
        }

        // End of EXPLORATION, restore the original letter in the board.
        this._board[row][col] = letter;

        // Optimization: incrementally remove the leaf nodes
        if (currNode.children.isEmpty()) {
            parent.children.remove(letter);
        }
    }
}

/*
lc 305

Time O(m*n+positions.length)  union find take nearly O(N)
when using path compression and union by rank
 */
class NumberOfIsland2 {

    private int[][] grid;
    private int count = 0;
    private int[] rowOffset = new int[]{0, 1, -1, 0};
    private int[] colOffset = new int[]{1, 0, 0, -1};

    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> res = new ArrayList<>();
        if (m == 0) {
            return res;
        }
        grid = new int[m][n];
        int[] root = new int[m * n];
        int[] weight = new int[m * n];
        //init union find array
        for (int i = 0; i < m * n; i++) {
            root[i] = i;
            weight[i] = 1;
        }
        //begin addLand
        for (int[] p : positions) {
            //if this point already is a land. skip
            if (grid[p[0]][p[1]] == 1) {
                res.add(count);
                continue;
            }
            count++;
            grid[p[0]][p[1]] = 1;
            for (int i = 0; i < 4; i++) {
                int newRow = p[0] + rowOffset[i];
                int newCol = p[1] + colOffset[i];
                //out of bound, skip
                if (!checkInGrid(newRow, newCol)) {
                    continue;
                }
                //if there is a land to connect
                if (grid[newRow][newCol] == 1) {
                    //find out whether it is already connected
                    //if so, do nothing, the new land just be a part of that island
                    //if not already connected, this connection would cause one less island in total
                    if (union(cordinateToId(p[0], p[1]), cordinateToId(newRow, newCol), root,
                        weight)) {
                        count--;
                    }
                }
            }
            res.add(count);
        }
        return res;
    }

    private int cordinateToId(int x, int y) {
        return x * grid[0].length + y;
    }

    private boolean checkInGrid(int x, int y) {
        return x >= 0 && x < grid.length && y >= 0 && y < grid[0].length;
    }

    private int find(int a, int[] root) {
        //while a is not a root
        while (a != root[a]) {
            //shorten the path to compress the tree.
            root[a] = root[root[a]];
            a = root[a];
        }
        return a;
    }

    private boolean union(int aId, int bId, int[] root, int[] weight) {
        int aRoot = find(aId, root);
        int bRoot = find(bId, root);
        //this two is already unioned.
        if (aRoot == bRoot) {
            return false;
        }
        //always use the heavy root to make sure
        if (weight[aRoot] > weight[bRoot]) {
            root[bRoot] = root[aRoot];
            weight[aRoot] += weight[bRoot];
        } else {
            root[aRoot] = root[bRoot];
            weight[bRoot] += root[aRoot];
        }
        return true;
    }

}

/*
    773. Sliding Puzzle
    Time O(N*M*(N*M)!)  there are possible (N*M)! state
    Space O(N*M*(N*M)!)
*/
class SlidingPuzzleMyOwn {

    public int slidingPuzzle(int[][] board) {
        //check if the board valid
        if (board == null || board.length == 0 || board[0].length == 0) {
            return -1;
        }

        int N = board.length;
        int M = board[0].length;

        //build the final state of this problem.
        int[][] targetBoard = new int[N][M];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                targetBoard[i][j] = i * M + j + 1;
            }
        }
        targetBoard[N - 1][M - 1] = 0;
        State targetState = new State(targetBoard, N - 1, M - 1);

        int currentSteps = -1;

        Set<State> visitedState = new HashSet<>();
        Queue<State> queue = new LinkedList();

        //build the original state
        State original = null;
        int[][] directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == 0) {
                    original = new State(deepCopyOfBoardArray(board), i, j);
                    break;
                }
            }
        }
        queue.offer(original);
        visitedState.add(original);

        while (!queue.isEmpty()) {
            int size = queue.size();
            //one more level bfs, means one more step.
            currentSteps++;
            for (int i = 0; i < size; i++) {
                State cur = queue.poll();
                if (targetState.equals(cur)) {
                    return currentSteps;
                }

                for (int j = 0; j < 4; j++) {
                    int[] direction = directions[j];
                    int nextRow = cur.posOf0[0] + direction[0];
                    int nextCol = cur.posOf0[1] + direction[1];
                    if (inBoard(nextRow, nextCol, cur.board)) {

                        int[][] newBoard = deepCopyOfBoardArray(cur.board);
                        //swap cur 0's position with new position.
                        int temp = newBoard[nextRow][nextCol];
                        newBoard[nextRow][nextCol] = newBoard[cur.posOf0[0]][cur.posOf0[1]];
                        newBoard[cur.posOf0[0]][cur.posOf0[1]] = temp;
                        State newState = new State(newBoard, nextRow, nextCol);

                        if (!visitedState.contains(newState)) {
                            queue.offer(newState);
                            visitedState.add(newState);
                        }
                    }
                }
            }
        }

        return -1;
    }

    private boolean inBoard(int row, int col, int[][] grid) {
        return row >= 0 && col >= 0 && row < grid.length && col < grid[0].length;
    }

    private int[][] deepCopyOfBoardArray(int[][] board) {
        int[][] res = new int[board.length][board[0].length];
        for (int i = 0; i < res.length; i++) {
            for (int j = 0; j < res[0].length; j++) {
                res[i][j] = board[i][j];
            }
        }
        return res;
    }


    private class State {

        int[][] board;
        int[] posOf0;

        State(int[][] board, int rowOf0, int colOf0) {
            this.board = board;
            this.posOf0 = new int[]{rowOf0, colOf0};
        }

        /*
             using its String format in hash function
         */
        @Override
        public int hashCode() {
            return toString().hashCode();
        }

        /*
            convert the 2d array to String.
         */
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[0].length; j++) {
                    sb.append(board[i][j]);
                    sb.append("|");
                }
            }
            return sb.toString();
        }

        /*
            Determine the equality by its String format.
         */
        @Override
        public boolean equals(Object obj) {
            return toString().equals(obj.toString());
        }
    }
}


//closed to constant time
class UnionFind {

    int[] idx;
    int[] weight;

    public UnionFind(int size) {
        idx = new int[size];
        weight = new int[size];
        for (int i = 0; i < size; i++) {
            idx[i] = i;
            weight[i] = 1;
        }
    }

    //avg  nearly to O(1)
    public int find(int a) {
        while (a != idx[a]) {
            idx[a] = idx[idx[a]];
            //compress the path
            a = idx[a];
        }
        return a;
    }

    public int getCount() {
        int count = 0;
        for (int i = 0; i < idx.length; i++) {
            if (i == idx[i]) {
                count++;
            }
        }
        return count;
    }

    //理论上，从一个完全不相连通的N个对象的集合开始，任意顺序的M次union()调用所需的时间为O(N+Mlg*N)。
    //其中lg*N称为迭代对数（iterated logarithm）。实数的迭代对数是指须对实数连续进行几次对数运算后，
    // 其结果才会小于等于1。这个函数增加非常缓慢，可以视为近似常数（例如2^65535的迭代对数为5）。
    // aveg inverse Ackermann function nearly O(1)
    /*
    Definition: A function of two parameters whose value grows very, very slowly.
     Formal Definition: α(m,n) = min{i≥ 1: A(i, ⌊ m/n⌋) > log2 n} where A(i,j)
     is Ackermann's function.
     */
    public boolean union(int a, int b) {
        int aRoot = find(a);
        int bRoot = find(b);
        //already unioned
        if (aRoot == bRoot) {
            return false;
        }
        if (weight[aRoot] > weight[bRoot]) {
            idx[bRoot] = idx[aRoot];
            weight[aRoot] += weight[bRoot];
        } else {
            idx[aRoot] = idx[bRoot];
            weight[bRoot] += weight[aRoot];
        }
        return true;
    }
}
