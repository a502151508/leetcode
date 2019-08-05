import java.awt.*;
import java.io.FileReader;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) {
        int[] nums = {5, 1, 1, 2, 5, 6, 3, 4, 76, 5, 3};
        for(int num:nums){
            System.out.println((num - 1) >>> 1);
        }
        findKthLargest(nums, 3);
    }

    public static int findKthLargest(int[] nums, int k) {

        PriorityQueue<Integer> pq = new PriorityQueue<>(k, Comparator.comparingInt(o -> o));
        for (int num : nums) {
            if (pq.size() == k) {
                if (num > pq.peek()) {
                    pq.poll();
                    pq.add(num);
                }
            } else
                pq.add(num);
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
        if (nums.length <= 1)
            return;
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
        if (intervals.length == 0)
            return 0;
        PriorityQueue<Integer> meetingQueue = new PriorityQueue<Integer>(intervals.length, Comparator.comparingInt(o -> o));
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
            if (m1[0] < m2[0])
                return -1;
            else if (m1[0] > m2[0])
                return 1;
            else {
                if (m1[1] >= m2[1])
                    return 1;
                else
                    return -1;
            }
        });
        for (int i = 0; i < meetings.size() - 1; i++) {
            int[] m1 = meetings.get(i);
            int[] m2 = meetings.get(i + 1);
            if (m1[1] > m2[0])
                return false;
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
        if (head == null)
            return null;
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
        target.get(target.size() - 1).random = randomIndex[target.size() - 1] != -1 ? target.get(randomIndex[target.size() - 1]) : null;
        return target.get(0);
    }

    public static boolean exist(char[][] board, String word) {
        int row = board.length;
        if (row == 0)
            return false;
        int col = board[0].length;
        if (row * col < word.length()) {
            return false;
        }
        char[] chars = word.toCharArray();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (chars[0] == board[i][j]) {
                    if (searchAround(i, j, chars, 1, board, null))
                        return true;
                }
            }
        }
        return false;
    }

    static boolean searchAround(int i, int j, char[] chars, int t, char[][] board, boolean[][] beenTo) {
        int row = board.length;
        int col = board[0].length;
        if (beenTo == null)
            beenTo = new boolean[row][col];
        beenTo[i][j] = true;
        if (t == chars.length)
            return true;
        if (i < row - 1 && board[i + 1][j] == chars[t] && !beenTo[i + 1][j] && searchAround(i + 1, j, chars, t + 1, board, beenTo))
            return true;
        if (i > 0 && board[i - 1][j] == chars[t] && !beenTo[i - 1][j] && searchAround(i - 1, j, chars, t + 1, board, beenTo)) {
            return true;
        }
        if (j > 0 && board[i][j - 1] == chars[t] && !beenTo[i][j - 1] && searchAround(i, j - 1, chars, t + 1, board, beenTo)) {
            return true;
        }
        if (j < col - 1 && board[i][j + 1] == chars[t] && !beenTo[i][j + 1] && searchAround(i, j + 1, chars, t + 1, board, beenTo)) {
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
            if (list == null) list = new ArrayList<String>();
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
        if (strs.length == 0)
            return new LinkedList<>();
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
        if (strs.length == 0)
            return new LinkedList<>();
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
        if (nums.length == 0)
            return -1;
        return findTarget(0, nums.length - 1, target, nums);
    }

    public static int findTarget(int start, int end, int target, int[] nums) {
        if (start == end || start + 1 == end) {
            if (nums[start] == target)
                return start;
            else if (nums[end] == target)
                return end;
            else return -1;
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
        } else
            return mid;
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
        Collections.sort(s, (s1, s2) -> {
            if (s1[0] == s2[0]) {
                return 0;
            } else if (s1[0] < s2[0]) {
                return -1;
            } else {
                return 1;
            }
        });
        for (int i = 0; i < s.size() - 1; i++) {
            int j = i + 1;
            if (canMerge(s.get(i), s.get(j))) {
                int[] newArray = {s.get(i)[0], s.get(j)[1] > s.get(i)[1] ? s.get(j)[1] : s.get(i)[1]};
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
        return (a[0] >= b[0] && a[0] <= b[1]) || (a[1] >= b[0] && a[1] <= b[1]) || (a[0] <= b[0] && a[1] >= b[0]) || (a[0] <= b[1] && a[1] >= b[1]);
    }

    public static int[][] kClosest(int[][] points, int K) {
        int[][] result = new int[K][2];
        divide(0, points.length, points);
        System.arraycopy(points, 0, result, 0, K);
        return result;
    }

    public static void divide(int i, int j, int[][] points) {
        Random r = new Random();
        int med = r.nextInt(points.length);
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
        if (row == 0)
            return 0;
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

    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode copyHead = head;
        while (l1.next == null && l2.next == null) {
            if (l1.val < l2.val || l2 == null) {
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
            if (stack.size() != 0 && !data.containsKey(stack.peek()))
                return false;
            if (c == data.get(stack.peek())) {
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
                    dis = Math.abs(nums[i] + nums[f] + nums[l] - target) < Math.abs(dis - target) ? nums[i] + nums[f] + nums[l] : dis;
                    f++;
                } else if (nums[i] + nums[f] + nums[l] - target > 0) {
                    dis = Math.abs(nums[i] + nums[f] + nums[l] - target) < Math.abs(dis - target) ? nums[i] + nums[f] + nums[l] : dis;
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
            if (nums[k] > 0) break;
            if (k > 0 && nums[k] == nums[k - 1]) continue;
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
        if (s.length() == 1)
            return 1;
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
            if (i > medIndex)
                break;
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
            if (i > medIndex)
                break;
            v1 = v2;
            v2 = nums1[t1];
            t1++;
            i++;
        }
        while (t2 < nums2.length) {
            if (i > medIndex)
                break;
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

    public String longestPalindrome(String s) {
        if (s == null || s.length() < 1) return "";
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
        if (numRows == 1 || s.length() < numRows)
            return s;
        else {
            int i = 0;
            int t = 0;
            StringBuilder[] results = new StringBuilder[numRows];
            boolean flag = true;
            while (t < s.length()) {
                if (results[i] == null)
                    results[i] = new StringBuilder();
                results[i].append(s.charAt(t));
                t++;
                if (flag)
                    i++;
                else
                    i--;
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
        if (str.length() == 0)
            return 0;
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
        if (i == 0)
            return 0;
        str = str.substring(0, i);
        if (str.equals("-") || str.equals("+"))
            return 0;
        int res;
        try {
            res = Integer.parseInt(str);
        } catch (Exception e) {
            if (sign)
                return Integer.MAX_VALUE;
            else
                return Integer.MIN_VALUE;
        }
        return res;
    }

    public boolean isMatch(String s, String p) {
        if (p.isEmpty())
            return s.isEmpty();
        int si = 0;
        int pi = 0;
        while (si < s.length() && pi < p.length()) {
            if (pi < p.length() - 1 && p.charAt(pi + 1) == '*') {
                if (s.charAt(si) == p.charAt(pi) || p.charAt(pi) == '.') {
                    si++;
                    continue;
                }
                char now = p.charAt(pi);
                pi = pi + 2;
                if (pi < p.length()) {
                    while (pi < p.length() && p.charAt(pi) == now) {
                        pi++;
                    }
                }

            } else {
                if (s.charAt(si) == p.charAt(pi)) {
                    si++;
                    pi++;
                } else if (p.charAt(pi) == '.') {
                    si++;
                    pi++;
                } else return false;
            }
        }
        return si == s.length() && (pi == p.length() || (pi + 2 == p.length() && p.charAt(p.length() - 1) == '*'));
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
        if (strs.length == 0)
            return "";
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

        if (cur == 0)
            return "";
        else return strs[0].substring(0, cur);
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