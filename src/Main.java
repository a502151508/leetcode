import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javafx.util.Pair;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        Main m = new Main();

    }


    //O(n(words)*m(ave letters))  O(Letters Numbers)
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
            boolean flag = false;
            TrieNodeArray curNode = root;
            for (char c : word.toCharArray()) {
                curNode = curNode.children[c - 'a'];
                count++;
                if (curNode != null && !word.equals(curNode.word)) {
                    if (curNode.word != null) {
                        flag = checkIfConcatenated(root, word.substring(count));
                    }
                    if (flag) {
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
        boolean flag = false;
        for (char c : word.toCharArray()) {
            if (node.children[c - 'a'] != null) {
                node = node.children[c - 'a'];
                count++;
                if (node.word != null) {
                    flag = checkIfConcatenated(root, word.substring(count));
                }
            } else {
                return false;
            }
            if (flag) {
                return true;
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


    public boolean searchPartitionK(int[] groups, int row, int[] nums, int target) {
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
            if (groups[i] == 0) {
                break;
            }
        }
        return false;
    }

    //平分成k组  回溯法暴力搜索
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = Arrays.stream(nums).sum();
        if (sum % k > 0) {
            return false;
        }
        int target = sum / k;

        Arrays.sort(nums);
        int row = nums.length - 1;
        if (nums[row] > target) {
            return false;
        }
        while (nums[row] == target) {
            row--;
            k--;
        }
        return searchPartitionK(new int[k], row, nums, target);
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
                    unionFind(i, j, root, ids);
                }
            }
        }
        return countSets(root);
    }

    public int mincostTickets(int[] days, int[] costs) {
        int[] dp = new int[days.length];
        int[] durations = new int[]{1, 7, 30};
        for (int i = dp.length - 1; i >= 0; i--) {
            int j = i;
            int min = Integer.MAX_VALUE;
            for (int d = 0; d < durations.length; d++) {
                while (j <= days.length - 1 && days[j] - days[i] < durations[d]) {
                    j++;
                }
                if (j == days.length) {
                    min = Math.min(min, costs[d]);
                } else {
                    min = Math.min(min, dp[j] + costs[d]);
                }
            }
            dp[i] = min;
        }
        return dp[0];
    }

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, -1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            int min = Integer.MAX_VALUE;
            boolean flag = false;
            for (int c : coins) {
                if (i - c >= 0 && dp[i - c] != -1) {
                    flag = true;
                    min = Math.min(min, dp[i - c] + 1);
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


    public boolean isHappy(int n) {
        Set<Integer> set = new HashSet<>();
        int m = 0;
        while (true) {
            while (n != 0) {
                m += Math.pow(n % 10, 2);
                n /= 10;
            }
            if (m == 1) {
                return true;
            }
            if (set.contains(m)) {
                return false;
            } else {
                set.add(m);
                n = m;
                m = 0;
            }
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

    public static boolean canFinish(int numCourses, int[][] prerequisites) {
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
        while (!readyToMove.isEmpty()) {
            int movedCourse = readyToMove.pop();
            count++;
            for (int[] pre : prerequisites) {
                if (pre[1] == movedCourse) {
                    if (--inDegrees[pre[0]] == 0) {
                        readyToMove.push(pre[0]);
                    }
                }
            }
        }
        return count == numCourses;
    }






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


    public static int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        if (image.length == 0) {
            return image;
        }
        fillFlood(image, sr, sc, image[sr][sc], newColor,
            new boolean[image.length][image[0].length]);
        return image;
    }

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
            char c1 = (char) (getRoot(c - 'a', root) + 'a');
            res.append(c1);
        }
        return res.toString();
    }

    private static void unionFindSmallDicString(int a, int b, int[] root) {
        int aRoot = getRoot(a, root);
        int bRoot = getRoot(b, root);
        if (aRoot == bRoot) {
            return;
        }
        if (root[bRoot] > root[aRoot]) {
            root[bRoot] = root[aRoot];
        } else {
            root[aRoot] = root[bRoot];
        }
    }


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
            cur.next = new ListNode(min.val);
            cur = cur.next;
            if (min.next != null) {
                pq.add(min.next);
            }
        }
        return head.next;
    }

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
                unionFind(e.getValue(), s.get(e.getKey() + 1), root, ids);
            }
            if (s.containsKey(e.getKey() - 1)) {
                unionFind(e.getValue(), s.get(e.getKey() - 1), root, ids);
            }
        }
        int max = Integer.MIN_VALUE;
        for (int id : ids) {
            max = Math.max(max, id);
        }
        return max;
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
            root[a] = root[root[a]];
            a = root[a];
        }
        return a;
    }

    private int countSets(int[] root) {
        int cnt = 0;
        for (int i = 0; i <= root.length - 1; i++) {
            if (getRoot(i, root) == i) {
                cnt++;
            }
        }
        return cnt;
    }


    public static int[] nextGreaterElements2(int[] nums) {
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
     * word ladder list 1.找到每个单词的转移分支，通过*来遮盖某一位来完成这个map 2。双向bfs，每个bfs节点保存当前的所有路径
     * 3.相遇后将当前单词列表加入答案，并完成本次队列里的遍历后结束 4
     */
    public static List<List<String>> findLadders(String beginWord, String endWord,
        List<String> wordList) {
        Set<List<String>> res = new HashSet<>();
        if (!wordList.contains(endWord)) {
            return new ArrayList<>(res);
        }
        if (beginWord.equals(endWord)) {
            List<String> singleResult = new ArrayList<>();
            singleResult.add(beginWord);
            res.add(singleResult);
            return new ArrayList<>(res);
        }

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

        Queue<Pair<String, List<String>>> qF = new LinkedList<>();
        Queue<Pair<String, List<String>>> qE = new LinkedList<>();
        Map<String, List<List<String>>> isVistedF = new HashMap<>();
        Map<String, List<List<String>>> isVistedE = new HashMap<>();
        int resSize = Integer.MAX_VALUE;
        qF.offer(new Pair<>(beginWord, new ArrayList<>(Arrays.asList(new String[]{beginWord}))));
        qE.offer(new Pair<>(endWord, new ArrayList<>(Arrays.asList(new String[]{endWord}))));
        List<List<String>> eVisited = new ArrayList<>();
        List<List<String>> fVisited = new ArrayList<>();
        eVisited.add(qE.peek().getValue());
        isVistedE.put(endWord, eVisited);
        fVisited.add(qF.peek().getValue());
        isVistedF.put(beginWord, fVisited);
        while (!qF.isEmpty() || !qE.isEmpty()) {
            if (!qF.isEmpty()) {
                Pair<String, List<String>> curPair = qF.remove();
                if (!(!res.isEmpty() && resSize <= curPair.getValue().size())) {
                    String curWord = curPair.getKey();
                    for (int i = 0; i < curPair.getKey().length(); i++) {
                        StringBuilder mask = new StringBuilder();
                        mask.append(curWord.substring(0, i));
                        mask.append("*");
                        mask.append(curWord.substring(i + 1));
                        List<String> wordListForAMask = comboDic
                            .getOrDefault(mask.toString(), new ArrayList<>());
                        for (String nextWord : wordListForAMask) {
                            if (nextWord.equals(endWord)) {
                                curPair.getValue().add(endWord);
                                List<String> newRes = new ArrayList<>(
                                    Arrays.asList(new String[curPair.getValue().size()]));
                                Collections.copy(newRes, curPair.getValue());
                                if (newRes.size() <= resSize) {
                                    res.add(newRes);
                                    resSize = newRes.size();
                                }
                            }
                            if (isVistedE.containsKey(nextWord)) {
                                List<String> newRes = new ArrayList<>();
                                newRes.addAll(curPair.getValue());
                                List<List<String>> anotherParts = isVistedE.get(nextWord);
                                for (List<String> anotherPart : anotherParts) {
                                    List<String> wholeRes = new ArrayList<>(newRes);
                                    wholeRes.addAll(anotherPart);
                                    if (wholeRes.size() <= resSize) {
                                        res.add(wholeRes);
                                        resSize = wholeRes.size();
                                    }
                                }
                            } else {
                                List<String> next = new ArrayList<>();
                                next.addAll(curPair.getValue());
                                next.add(nextWord);
                                List<List<String>> allPath = isVistedF
                                    .getOrDefault(nextWord, new ArrayList<>());
                                allPath.add(next);
                                isVistedF.put(nextWord, allPath);
                                qF.offer(new Pair<>(nextWord, next));
                            }
                        }
                    }
                }
            }
            if (!qE.isEmpty()) {
                Pair<String, List<String>> curPair = qE.remove();
                if (!(!res.isEmpty() && resSize <= curPair.getValue().size())) {
                    String curWord = curPair.getKey();
                    for (int i = 0; i < curPair.getKey().length(); i++) {
                        StringBuilder mask = new StringBuilder();
                        mask.append(curWord.substring(0, i));
                        mask.append("*");
                        mask.append(curWord.substring(i + 1));
                        List<String> wordListForAMask = comboDic
                            .getOrDefault(mask.toString(), new ArrayList<>());
                        for (String nextWord : wordListForAMask) {
                            if (nextWord.equals(endWord)) {
                                curPair.getValue().add(endWord);
                                List<String> newRes = new ArrayList<>(
                                    Arrays.asList(new String[curPair.getValue().size()]));
                                Collections.copy(newRes, curPair.getValue());
                                if (newRes.size() <= resSize) {
                                    res.add(newRes);
                                    resSize = newRes.size();
                                }
                            }
                            if (isVistedF.containsKey(nextWord)) {
                                List<String> newRes = new ArrayList<>(curPair.getValue());
                                List<List<String>> anotherParts = isVistedF.get(nextWord);
                                for (List<String> anotherPart : anotherParts) {
                                    List<String> wholeRes = new ArrayList<>(anotherPart);
                                    wholeRes.addAll(newRes);
                                    if (wholeRes.size() <= resSize) {
                                        res.add(wholeRes);
                                        resSize = wholeRes.size();
                                    }
                                }
                            } else {
                                List<String> next = new LinkedList<>();
                                next.addAll(curPair.getValue());
                                next.add(0, nextWord);
                                List<List<String>> allPath = isVistedF
                                    .getOrDefault(nextWord, new ArrayList<>());
                                allPath.add(next);
                                isVistedE.put(nextWord, allPath);
                                qE.offer(new Pair<>(nextWord, next));
                            }
                        }
                    }
                }

            }
        }
        return new ArrayList<>(res);
    }


    public List<List<String>> findLaddersZUIJIA(String beginWord, String endWord,
        List<String> wordList) {
        List<List<String>> ans = new ArrayList<>();
        if (!wordList.contains(endWord)) {
            return ans;
        }
        // 利用 BFS 得到所有的邻居节点
        HashMap<String, ArrayList<String>> map = new HashMap<>();
        bfs(beginWord, endWord, wordList, map);
        ArrayList<String> temp = new ArrayList<String>();
        // temp 用来保存当前的路径
        temp.add(beginWord);
        findLaddersHelper(beginWord, endWord, map, temp, ans);
        return ans;
    }

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

    //利用递归实现了双向搜索
    private void bfs(String beginWord, String endWord, List<String> wordList,
        HashMap<String, ArrayList<String>> map) {
        Set<String> set1 = new HashSet<String>();
        set1.add(beginWord);
        Set<String> set2 = new HashSet<String>();
        set2.add(endWord);
        Set<String> wordSet = new HashSet<String>(wordList);
        bfsHelper(set1, set2, wordSet, true, map);
    }

    // direction 为 true 代表向下扩展，false 代表向上扩展
    private boolean bfsHelper(Set<String> set1, Set<String> set2, Set<String> wordSet,
        boolean direction,
        HashMap<String, ArrayList<String>> map) {
        //set1 为空了，就直接结束
        //比如下边的例子就会造成 set1 为空
    /*	"hot"
		"dog"
		["hot","dog"]*/
        if (set1.isEmpty()) {
            return false;
        }
        // set1 的数量多，就反向扩展
        if (set1.size() > set2.size()) {
            return bfsHelper(set2, set1, wordSet, !direction, map);
        }
        // 将已经访问过单词删除
        wordSet.removeAll(set1);
        wordSet.removeAll(set2);

        boolean done = false;

        // 保存新扩展得到的节点
        Set<String> set = new HashSet<String>();

        for (String str : set1) {
            //遍历每一位
            for (int i = 0; i < str.length(); i++) {
                char[] chars = str.toCharArray();

                // 尝试所有字母
                for (char ch = 'a'; ch <= 'z'; ch++) {
                    if (chars[i] == ch) {
                        continue;
                    }
                    chars[i] = ch;

                    String word = new String(chars);

                    // 根据方向得到 map 的 key 和 val
                    String key = direction ? str : word;
                    String val = direction ? word : str;

                    ArrayList<String> list =
                        map.containsKey(key) ? map.get(key) : new ArrayList<String>();

                    //如果相遇了就保存结果
                    if (set2.contains(word)) {
                        done = true;
                        list.add(val);
                        map.put(key, list);
                    }

                    //如果还没有相遇，并且新的单词在 word 中，那么就加到 set 中
                    if (!done && wordSet.contains(word)) {
                        set.add(word);
                        list.add(val);
                        map.put(key, list);
                    }
                }
            }
        }

        //一般情况下新扩展的元素会多一些，所以我们下次反方向扩展  set2
        return done || bfsHelper(set2, set, wordSet, !direction, map);

    }

    //word ladder length
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

    //Accepted
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

    //time exceeded
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
    public int maxProfit3(int K, int[] prices) {
        if (prices.length == 0) {
            return 0;
        }
        if (K >= prices.length / 2) {
            return maxProfit2(prices);
        }
        int[][] dp = new int[prices.length][K + 1];
        for (int k = 1; k <= K; k++) {
            int min = prices[0];
            for (int i = 1; i < prices.length; i++) {
                //找出第 1 天到第 i 天 prices[buy] - dp[buy][k - 1] 的最小值
                //dp[i][k]应为求 price[i]-price[j]+dp[j][k-1]的最大值即为最后一次买入
                //在j天，并与不买入的数值比较，也就是dp[i-1][k].
                //转化为求price[j]-dp[j][k-1]的最小值
                //price[j]-dp[j][k-1] 其中j的取值为0~i，所以可用一个min储存起来，以i遍历累积过去。
                min = Math.min(prices[i] - dp[i][k - 1], min);
                //比较不操作和选择一天买入的哪个值更大
                dp[i][k] = Math.max(dp[i - 1][k], prices[i] - min);
            }
        }
        return dp[prices.length - 1][K];
    }

    //sell stock 1， only allow do one transaction
    public static int maxProfit1(int[] prices) {
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

    //数字个位在最后
    public static ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
        Stack<Integer> s1 = new Stack<Integer>();
        Stack<Integer> s2 = new Stack<Integer>();

        while (l1 != null) {
            s1.push(l1.val);
            l1 = l1.next;
        }
        ;
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

    static int getLength(ListNode l1) {
        int res = 0;
        while (l1 != null) {
            res++;
            l1 = l1.next;
        }
        return res;
    }

    //数字个位在链表第一个
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

    public Node() {
    }

    public Node(int _val, Node _next, Node _random) {
        val = _val;
        next = _next;
        random = _random;
    }
}

class MinStack {

    // 数据栈
    private Stack<Integer> data;
    // 辅助栈
    private Stack<Integer> helper;

    /**
     * initialize your data structure here.
     */
    public MinStack() {
        data = new Stack<>();
        helper = new Stack<>();
    }

    // 思路 2：辅助栈和数据栈不同步
    // 关键 1：辅助栈的元素空的时候，必须放入新进来的数
    // 关键 2：新来的数小于或者等于辅助栈栈顶元素的时候，才放入（特别注意这里等于要考虑进去）
    // 关键 3：出栈的时候，辅助栈的栈顶元素等于数据栈的栈顶元素，才出栈，即"出栈保持同步"就可以了

    public void push(int x) {
        // 辅助栈在必要的时候才增加
        data.add(x);
        // 关键 1 和 关键 2
        if (helper.isEmpty() || helper.peek() >= x) {
            helper.add(x);
        }
    }

    public void pop() {
        // 关键 3：data 一定得 pop()
        if (!data.isEmpty()) {
            // 注意：声明成 int 类型，这里完成了自动拆箱，从 Integer 转成了 int，因此下面的比较可以使用 "==" 运算符
            // 参考资料：https://www.cnblogs.com/GuoYaxiang/p/6931264.html
            // 如果把 top 变量声明成 Integer 类型，下面的比较就得使用 equals 方法
            int top = data.pop();
            if (top == helper.peek()) {
                helper.pop();
            }
        }
    }

    public int top() {
        if (!data.isEmpty()) {
            return data.peek();
        }
        throw new RuntimeException("栈中元素为空，此操作非法");
    }

    public int getMin() {
        if (!helper.isEmpty()) {
            return helper.peek();
        }
        throw new RuntimeException("栈中元素为空，此操作非法");
    }

}

//Not allow two events at the same time
class MyCalendar {

    TreeMap<Integer, Integer> tm;

    public MyCalendar() {
        tm = new TreeMap<>();
    }

    public boolean book(int start, int end) {
        if (end <= start) {
            return false;
        }
        Map.Entry<Integer, Integer> low = tm.floorEntry(start);
        Map.Entry<Integer, Integer> high = tm.ceilingEntry(start);
        if (low == null && high == null) {
            tm.put(start, end);
            return true;
        }
        if (low != null && low.getValue() > start) {
            return false;
        }
        if (high != null && end > high.getKey()) {
            return false;
        }
        tm.put(start, end);
        return true;
    }
}

//not allow three events at the same time
class MyCalendarTwo {

    List<int[]> calendar;
    List<int[]> overlaps;

    MyCalendarTwo() {
        calendar = new ArrayList();
        overlaps = new ArrayList<>();
    }

    public boolean book(int start, int end) {
        for (int[] iv : overlaps) {
            if (iv[0] < end && start < iv[1]) {
                return false;
            }
        }
        for (int[] iv : calendar) {
            if (iv[0] < end && start < iv[1]) {
                overlaps.add(new int[]{Math.max(start, iv[0]), Math.min(end, iv[1])});
            }
        }
        calendar.add(new int[]{start, end});
        return true;
    }
}

//Return Largest num of events that hold together
class MyCalendarThree {

    TreeMap<Integer, Integer> delta;

    public MyCalendarThree() {
        delta = new TreeMap();
    }

    public int book(int start, int end) {
        delta.put(start, delta.getOrDefault(start, 0) + 1);
        delta.put(end, delta.getOrDefault(end, 0) - 1);

        int active = 0, ans = 0;
        for (int d : delta.values()) {
            active += d;
            if (active > ans) {
                ans = active;
            }
        }
        return ans;
    }
}

class RandomizedSet {

    Map<Integer, Integer> dict;
    List<Integer> list;
    Random rand = new Random();

    /**
     * Initialize your data structure here.
     */
    public RandomizedSet() {
        dict = new HashMap();
        list = new ArrayList();
    }

    /**
     * Inserts a value to the set. Returns true if the set did not already contain the specified
     * element.
     */
    public boolean insert(int val) {
        if (dict.containsKey(val)) {
            return false;
        }

        dict.put(val, list.size());
        list.add(val);
        return true;
    }

    /**
     * Removes a value from the set. Returns true if the set contained the specified element.
     */
    public boolean remove(int val) {
        if (!dict.containsKey(val)) {
            return false;
        }

        // move the last element to the place idx of the element to delete
        int lastElement = list.get(list.size() - 1);
        int idx = dict.get(val);
        list.set(idx, lastElement);
        dict.put(lastElement, idx);
        // delete the last element
        list.remove(list.size() - 1);
        dict.remove(val);
        return true;
    }

    /**
     * Get a random element from the set.
     */
    public int getRandom() {
        return list.get(rand.nextInt(list.size()));
    }
}

class ShuffleArray {

    int[] nums;
    int[] originNums;

    Random rand = new Random();

    private int randRange(int min, int max) {
        return rand.nextInt(max - min) + min;
    }

    private void swapAt(int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public ShuffleArray(int[] nums) {
        this.nums = nums;
        this.originNums = Arrays.copyOf(nums, nums.length);
    }

    public int[] reset() {
        nums = Arrays.copyOf(originNums, nums.length);
        return nums;
    }

    public int[] shuffle() {
        for (int i = 0; i < nums.length; i++) {
            swapAt(i, randRange(i, nums.length));
        }
        return nums;
    }
}

class MyStack {

    private Queue<Integer> q1;

    /**
     * Initialize your data structure here.
     */
    public MyStack() {
        q1 = new LinkedList<>();
    }

    /**
     * Push element x onto stack.
     */
    public void push(int x) {
        q1.add(x);
        int sz = q1.size();
        while (sz > 1) {
            q1.add(q1.remove());
            sz--;
        }
    }

    /**
     * Removes the element on top of the stack and returns that element.
     */
    public int pop() {
        return q1.remove();
    }

    /**
     * Get the top element.
     */
    public int top() {
        return q1.peek();
    }

    /**
     * Returns whether the stack is empty.
     */
    public boolean empty() {
        return q1.isEmpty();
    }
}

class MyQueue {


    private Stack<Integer> s1 = new Stack<>();
    private Stack<Integer> s2 = new Stack<>();
    int front;

    /**
     * Initialize your data structure here.
     */
    public MyQueue() {

    }

    /**
     * Push element x to the back of queue.
     */
    public void push(int x) {
        if (s1.empty()) {
            front = x;
        }
        s1.push(x);
    }

    /**
     * Removes the element from in front of queue and returns that element.
     */
    public int pop() {
        if (s2.isEmpty()) {
            while (!s1.isEmpty()) {
                s2.push(s1.pop());
            }
        }
        return s2.pop();
    }

    /**
     * Get the front element.
     */
    public int peek() {
        if (!s2.isEmpty()) {
            return s2.peek();
        }
        return front;
    }

    /**
     * Returns whether the queue is empty.
     */
    public boolean empty() {
        return s1.isEmpty() && s2.isEmpty();
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

//存的时候带时间，取的时候返回一个刚好小于等于这个取的时间的值
class TimeMap {

    /**
     * Initialize your data structure here.
     */
    Map<String, TreeMap<Integer, String>> map;

    public TimeMap() {
        map = new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
        if (!map.containsKey(key)) {
            map.put(key, new TreeMap());
        }
        TreeMap<Integer, String> treeMap = map.get(key);
        treeMap.put(timestamp, value);
    }

    public String get(String key, int timestamp) {
        TreeMap<Integer, String> treeMap = map.get(key);
        Integer tmKey = treeMap.floorKey(timestamp);
        if (tmKey == null) {
            return "";
        } else {
            return treeMap.get(tmKey);
        }
    }
}

class Logger {

    private Map<String, Integer> map;

    /**
     * Initialize your data structure here.
     */
    public Logger() {
        map = new HashMap<>();
    }

    /**
     * Returns true if the message should be printed in the given timestamp, otherwise returns
     * false. If this method returns false, the message will not be printed. The timestamp is in
     * seconds granularity.
     */
    public boolean shouldPrintMessage(int timestamp, String message) {
        if (map.containsKey(message)) {
            if (timestamp - map.get(message) >= 10) {
                map.put(message, timestamp);
                return true;
            }
            return false;
        } else {
            map.put(message, timestamp);
            return true;
        }
    }
}
