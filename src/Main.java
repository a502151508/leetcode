import java.beans.PropertyVetoException;
import java.util.concurrent.ConcurrentHashMap;
import javafx.util.Pair;
import java.util.*;
import java.util.List;
import sun.text.normalizer.Trie;

public class Main {

    public static void main(String[] args) {
        System.out.println(waterPlants(new int[]{2, 4, 5, 4, 1, 2}, 5, 7));
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

    public int[][] kClosest(int[][] points, int K) {
        int[][] res = new int[K][2];
        PriorityQueue<int[]> queue = new PriorityQueue<>(K, (t1, t2) -> dis(t1) - dis(t2));//构建一个小根堆
        for (int i = 0; i < points.length; i++) {
            queue.offer(points[i]); //将所有元素入队
        }

        for (int i = 0; i < K; i++) {
            res[i] = queue.poll(); //前k个元素出堆，就是我们所需要的结果。
        }

        return res;
    }

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


    public String fractionToDecimal(int numerator, int denominator) {
        StringBuilder res = new StringBuilder();
        if (numerator == 0) {
            return "0";
        }
        //if denominator == 0

        if (numerator < 0 ^ denominator < 0) {
            res.append('-');
        }
        long n = Math.abs(Long.valueOf(numerator));
        long d = Math.abs(Long.valueOf(denominator));
        long quo = n / d;
        long remainder = n % d;
        if (remainder == 0) {
            res.append(quo);
            return res.toString();
        } else {
            res.append(quo);
            res.append('.');
            Map<Long, Integer> map = new HashMap<>();
            while (remainder != 0) {
                quo = remainder * 10 / d;
                if (map.containsKey(remainder)) {
                    res.insert(map.get(remainder), "(");
                    res.append(")");
                    break;
                }
                res.append(quo);
                map.put(remainder, res.length() - 1);
                remainder = remainder * 10 % d;

            }
        }
        return res.toString();
    }

    /**
     * Morris中序遍历二叉树(node每次往右移之前打印节点)
     */
    public static void morrisIn(TreeNode head) {
        if (head == null) {
            return;
        }
        TreeNode cur = head;
        TreeNode prev = null;
        while (cur != null) {
            //没有左节点 直接去右节点
            if (cur.left == null) {
                System.out.println(cur.val);
                cur = cur.right;
            } else {
                //找先驱节点，即为中序遍历之前的最后一个节点
                prev = cur.left;
                while (prev.right != null && prev.right != cur) {
                    prev = prev.right;
                }
                if (prev.right == null) {
                    prev.right = cur;
                    cur = cur.left;
                } else {
                    prev.right = null;
                    System.out.println(cur.val);
                    cur = cur.right;
                }
            }
        }
    }

    //  非递归中根遍历
    public static void iterativeInOrder(TreeNode p) {
        if (p == null) {
            return;
        }
        Deque<TreeNode> stack = new ArrayDeque<>();
        while (stack.size() != 0 || p != null) {
            while (p != null) {
                stack.addLast(p);
                p = p.left;
            }
            p = stack.removeLast();
            System.out.println(p.val);
            p = p.right;
        }
    }

    //**********非递归的分层输出的层次遍历**********
    public static void iterativeLevelOrder_1(TreeNode p) {
        if (p == null) {
            return;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(p);
        while (!queue.isEmpty()) {
            int levelNum = queue.size();
            for (int i = 0; i < levelNum; i++) {
                p = queue.poll();
                if (p.left != null) {
                    queue.offer(p.left);
                }
                if (p.right != null) {
                    queue.offer(p.right);
                }
                System.out.println(p.val);
            }

        }
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
                unionFind(j, i, root, ids);
            }
        }

        List<Integer> res = new ArrayList<>(originCities.size());
        Iterator<Integer> itSrc = originCities.iterator();
        Iterator<Integer> itDest = destinationCities.iterator();

        while (itSrc.hasNext() && itDest.hasNext()) {
            res.add(getRoot(itSrc.next(), root) == getRoot(itDest.next(), root) ? 1 : 0);
        }

        return res;
    }

    private static void unionFind(int a, int b, int[] root, int[] ids) {
        int aRoot = getRoot(a, root);
        int bRoot = getRoot(b, root);
        if (aRoot == bRoot) {
            return;
        }

        if (ids[aRoot] < ids[bRoot]) {
            root[aRoot] = root[bRoot];
            ids[bRoot] += ids[aRoot];
        } else {
            root[bRoot] = root[aRoot];
            ids[aRoot] += ids[bRoot];
        }
    }

    private static int getRoot(int a, int[] root) {
        while (a != root[a]) {
            a = root[a];
        }
        return a;
    }


    public static int[] nextGreaterElements1(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
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

    public int[] nextGreaterElement(int[] findNums, int[] nums) {
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

    public int[] nextGreaterElements(int[] nums) {
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

    public static void moveZeroes(int[] nums) {
        int i = 0;
        int j = nums.length - 1;
        if (j < 0) {
            return;
        }
        while (i < j) {
            while (j > 0 && nums[j] == 0) {
                j--;
            }
            if (nums[i] != 0) {
                i++;
                continue;
            }
            swapIJ(nums, i, j);
            j--;
        }
    }

    static void swapIJ(int[] nums, int i, int j) {
        for (int t = i; t < j; t++) {
            nums[t] = nums[t + 1];
        }
        nums[j] = 0;
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

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new LinkedList<>();
        findResult(candidates, target, new Stack<>(), res, 0);
        return res;
    }

    static void findResult(int[] candidates, int target, Stack<Integer> nums,
        List<List<Integer>> res, int start) {
        if (target < 0) {
            return;
        }
        if (target == 0) {
            res.add(new LinkedList(nums));
        } else {
            for (int i = start; i < candidates.length && target - candidates[i] >= 0; i++) {
                nums.add(candidates[i]);
                findResult(candidates, target - candidates[i], nums, res, i);
                nums.pop();
            }
        }
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

    public static int removeDuplicates(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return 1;
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
                    mask.append(curWord.substring(0, i));
                    mask.append("*");
                    mask.append(curWord.substring(i + 1));
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

    public static Node copyRandomList(Node head) {
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

    public static List<List<String>> groupAnagrams3(String[] strs) {

        HashMap<String, ArrayList<String>> map = new HashMap();
        for (int i = 0; i < strs.length; i++) {
            String valKey = createAngVal(strs[i]);
            ArrayList<String> list = map.get(valKey);
            if (list == null) {
                list = new ArrayList<String>();
            }
            list.add(strs[i]);
            map.put(valKey, list);
        }
        return new ArrayList(map.values());
    }

    public static String createAngVal(String str) {
        char arr[] = new char[26];
        for (int i = 0; i < str.length(); i++) {
            ++arr[str.charAt(i) - 'a'];
        }
        return new String(arr);
    }

    public static List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new LinkedList<>();
        if (strs.length == 0) {
            return new LinkedList<>();
        }
        Map<Integer, List<String>> countGroup = new HashMap();
        for (String str : strs) {
            if (countGroup.containsKey(str.length())) {
                countGroup.get(str.length()).add(str);
            } else {
                LinkedList<String> l = new LinkedList<>();
                l.add(str);
                countGroup.put(str.length(), l);
            }
        }
        for (Map.Entry<Integer, List<String>> e : countGroup.entrySet()) {
            List<String> cur = e.getValue();
            while (cur.size() != 0) {
                List<String> singleResult = new LinkedList();
                String s = cur.get(0);
                String copyS = s;
                singleResult.add(s);
                cur.remove(0);
                outer:
                for (int i = 0; i < cur.size(); i++) {
                    for (int j = 0; j < cur.get(i).length(); j++) {
                        if (!copyS.contains(String.valueOf(cur.get(i).charAt(j)))) {
                            copyS = s;
                            continue outer;
                        } else {
                            int index = copyS.indexOf(cur.get(i).charAt(j));
                            copyS = copyS.substring(0, index) + copyS.substring(index + 1);
                        }
                    }
                    singleResult.add(cur.get(i));
                    cur.remove(i);
                    i--;
                    copyS = s;
                }
                result.add(singleResult);
            }
            if (cur.size() != 0) {
                List<String> singleResult = new LinkedList();
                singleResult.add(cur.get(0));
                result.add(singleResult);
            }
        }
        return result;
    }

    public static List<List<String>> groupAnagrams1(String[] strs) {
        List<List<String>> result = new LinkedList<>();
        if (strs.length == 0) {
            return new LinkedList<>();
        }
        Map<Integer, List<String>> countGroup = new HashMap();
        for (String str : strs) {
            if (countGroup.containsKey(str.length())) {
                countGroup.get(str.length()).add(str);
            } else {
                LinkedList<String> l = new LinkedList<>();
                l.add(str);
                countGroup.put(str.length(), l);
            }
        }

        for (Map.Entry<Integer, List<String>> e : countGroup.entrySet()) {
            List<String> cur = e.getValue();
            while (cur.size() != 0) {
                List<String> singleResult = new LinkedList();
                String s = cur.get(0);
                singleResult.add(s);
                cur.remove(0);
                outer:
                for (int i = 0; i < cur.size(); i++) {
                    for (int j = 0; j < s.length(); j++) {
                        if (!cur.get(i).contains(String.valueOf(s.charAt(j)))) {
                            break outer;
                        }
                    }
                    singleResult.add(cur.get(i));
                    cur.remove(i);
                }
                result.add(singleResult);
            }
            if (cur.size() != 0) {
                List<String> singleResult = new LinkedList();
                singleResult.add(cur.get(0));
                result.add(singleResult);
            }
        }
        return result;
    }

    public static int search(int[] nums, int target) {
        if (nums.length == 0) {
            return -1;
        }
        return findTarget(0, nums.length - 1, target, nums);
    }

    public static int findTarget(int start, int end, int target, int[] nums) {
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
        if (nums[mid] < target) {
            if (nums[end] >= target) {
                return findTarget(mid, end, target, nums);
            } else {
                int res = findTarget(start, mid, target, nums);
                return res != -1 ? res : findTarget(mid, end, target, nums);
            }

        } else if (nums[mid] > target) {
            if (nums[start] <= target) {
                return findTarget(start, mid, target, nums);
            } else {
                int res = findTarget(mid, end, target, nums);
                return res != -1 ? res : findTarget(start, mid, target, nums);
            }
        } else {
            return mid;
        }
    }

    public static int maxProfit(int[] prices) {
        int buyIn = Integer.MAX_VALUE;
        int sellOut = Integer.MIN_VALUE;
        int maxPro = Integer.MIN_VALUE;
        for (int i : prices) {
            if (i < buyIn) {
                buyIn = i;
                sellOut = Integer.MIN_VALUE;
            }
            sellOut = i > sellOut ? i : sellOut;
            maxPro = maxPro > sellOut - buyIn ? maxPro : sellOut - buyIn;
        }
        return maxPro;
    }

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

    public static boolean canMerge(int[] a, int[] b) {
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

    public static int numIslands(char[][] grid) {
        int row = grid.length;
        if (row == 0) {
            return 0;
        }
        int col = grid[0].length;
        int result = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '1') {
                    findIsland(grid, i, j, row, col);
                    result++;
                }
            }
        }
        return result;
    }

    public static void findIsland(char[][] grid, int row, int col, int rowM, int colM) {
        if (row < rowM - 1 && grid[row + 1][col] == '1') {
            grid[row + 1][col] = '0';
            findIsland(grid, row + 1, col, rowM, colM);
        }
        if (col < colM - 1 && grid[row][col + 1] == '1') {
            grid[row][col + 1] = '0';
            findIsland(grid, row, col + 1, rowM, colM);
        }
        if (row > 0 && grid[row - 1][col] == '1') {
            grid[row - 1][col] = '0';
            findIsland(grid, row - 1, col, rowM, colM);
        }
        if (col > 0 && grid[row][col - 1] == '1') {
            grid[row][col - 1] = '0';
            findIsland(grid, row, col - 1, rowM, colM);
        }
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode copyHead = head;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                copyHead.next = l1;
                l1 = l1.next;
            } else {
                copyHead.next = l2;
                l2 = l2.next;
            }
            copyHead = copyHead.next;
        }
        copyHead.next = l1 == null ? l2 : l1;
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

    public static int lengthOfLongestSubstring(String s) {
        LinkedHashSet<Character> hs = new LinkedHashSet<Character>();
        int max = 0;
        if (s.length() == 1) {
            return 1;
        }
        for (int i = 0; i < s.length() - 1; i++) {
            hs.add(s.charAt(i));
            for (int j = i + 1; j < s.length(); j++) {
                if (!hs.contains(s.charAt(j))) {
                    hs.add(s.charAt(j));
                } else {
                    max = max < hs.size() ? hs.size() : max;
                    hs.remove(s.charAt(i));
                    break;
                }
                max = max < hs.size() ? hs.size() : max;
            }
        }
        return max;
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

    public static int myAtoi(String str) {
        str = str.trim();
        boolean sign = true;
        int i = 0;
        if (str.length() == 0) {
            return 0;
        }
        if ("-".indexOf(str.charAt(0)) >= 0) {
            sign = false;
            i++;
        }
        if ("+".indexOf(str.charAt(0)) >= 0) {
            sign = true;
            i++;
        }
        for (; i < str.length(); i++) {
            if ("1234567890".indexOf(str.charAt(i)) < 0) {
                break;
            }
        }
        if (i == 0) {
            return 0;
        }
        str = str.substring(0, i);
        if (str.equals("-") || str.equals("+")) {
            return 0;
        }
        int res;
        try {
            res = Integer.parseInt(str);
        } catch (Exception e) {
            if (sign) {
                return Integer.MAX_VALUE;
            } else {
                return Integer.MIN_VALUE;
            }
        }
        return res;
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

class TreeNode {

    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int x) {
        val = x;
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

    public Node() {
    }

    public Node(int _val, Node _next, Node _random) {
        val = _val;
        next = _next;
        random = _random;
    }
}

class LRUCache extends LinkedHashMap<Integer, Integer> {

    private int capacity;

    public LRUCache(int capacity) {
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity;
    }
}