import java.util.*;

public class Main {

    public static void main(String[] args) {
        String[] s = {};
        System.out.println(longestCommonPrefix(s));
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

}


class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
    }
}