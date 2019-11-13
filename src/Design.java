import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Stack;
import java.util.TreeMap;
import java.util.concurrent.LinkedBlockingQueue;
import sun.net.www.protocol.http.HttpURLConnection.TunnelState;

/**
 * Created by Sichi Zhang on 2019/11/10.
 */
public class Design {

}

class LRUCache {


    Map<Integer, Integer> map;
    LinkedList<Integer> dl;
    int capacity;

    // first element is the least recently used one.
    public LRUCache(int capacity) {
        map = new HashMap<>();
        dl = new LinkedList<>();
        this.capacity = capacity;
    }

    //remove it and add it to last
    public int get(Integer key) {
        Integer res = map.getOrDefault(key, -1);
        if (res == -1) {
            return -1;
        }
        dl.remove(key);
        dl.addLast(key);
        return res;
    }

    //if reach capacity, remove the first node. then add the key in to last.
    public void put(Integer key, int value) {
        if (map.getOrDefault(key, -1) != -1) {
            dl.remove(key);
        }
        map.put(key, value);
        dl.offerLast(key);
        if (dl.size() > capacity) {
            map.remove(dl.removeFirst());
        }
    }

    public static void main(String[] args) {
        LRUCache lru = new LRUCache(2);
        lru.put(2, 1);
        lru.put(1, 1);
        lru.put(2, 3);
        lru.put(4, 1);
        int n1 = lru.get(2);
        lru.put(4, 1);
        int n2 = lru.get(1);
        int n3 = lru.get(2);
        System.out.println();
    }
}

class LRUCache1 {

    class DLinkedNode {

        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;
    }

    private void addNode(DLinkedNode node) {
        /*
          Always add the new node right after head.
         */
        node.prev = head;
        node.next = head.next;

        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node) {
        /*
          Remove an existing node from the linked list.
         */
        DLinkedNode prev = node.prev;
        DLinkedNode next = node.next;

        prev.next = next;
        next.prev = prev;
    }

    private void moveToHead(DLinkedNode node) {
        /*
          Move certain node in between to the head.
         */
        removeNode(node);
        addNode(node);
    }

    private DLinkedNode popTail() {
        /*
          Pop the current tail.
         */
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }

    private Map<Integer, DLinkedNode> cache = new HashMap<>();
    private int size;
    private int capacity;
    private DLinkedNode head, tail;

    public LRUCache1(int capacity) {
        this.size = 0;
        this.capacity = capacity;

        head = new DLinkedNode();
        // head.prev = null;

        tail = new DLinkedNode();
        // tail.next = null;

        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            return -1;
        }

        // move the accessed node to the head;
        moveToHead(node);

        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);

        if (node == null) {
            DLinkedNode newNode = new DLinkedNode();
            newNode.key = key;
            newNode.value = value;

            cache.put(key, newNode);
            addNode(newNode);

            ++size;

            if (size > capacity) {
                // pop the tail
                DLinkedNode tail = popTail();
                cache.remove(tail.key);
                --size;
            }
        } else {
            // update the value.
            node.value = value;
            moveToHead(node);
        }
    }
}

class LRUCache2 extends LinkedHashMap<Integer, Integer> {

    private int capacity;

    public LRUCache2(int capacity) {
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

class LFUCache {

    private int capacity;
    private Map<Integer, Integer> freqMap;
    private Map<Integer, Integer> map;
    private Map<Integer, LinkedList<Integer>> freqGroup;
    private int leastFreq = 1;

    public LFUCache(int capacity) {
        this.capacity = capacity;
        freqMap = new HashMap<>();
        map = new HashMap<>();
        freqGroup = new HashMap<>();
    }

    public int get(Integer key) {
        int res = map.getOrDefault(key, -1);
        if (res != -1) {
            int freq = freqMap.get(key);
            LinkedList<Integer> currentGroup = freqGroup.get(freq);
            currentGroup.remove(key);
            freq++;
            freqMap.put(key, freq);
            LinkedList<Integer> freqList = freqGroup.getOrDefault(freq, new LinkedList<>());
            freqList.addLast(key);
            freqGroup.put(freq, freqList);
        }
        return res;
    }

    public void put(Integer key, int value) {
        if (capacity == 0) {
            return;
        }
        if (map.size() == capacity && !map.containsKey(key)) {
            LinkedList<Integer> leastFreqGroup = freqGroup.get(leastFreq);
            while (leastFreqGroup == null || leastFreqGroup.size() == 0) {
                leastFreq++;
                leastFreqGroup = freqGroup.getOrDefault(leastFreq, null);
            }
            int removedKey = leastFreqGroup.removeFirst();
            freqMap.remove(removedKey);
            map.remove(removedKey);
        }

        int freq = freqMap.getOrDefault(key, 0);
        if (freq > 0) {
            LinkedList<Integer> currentGroup = freqGroup.get(freq);
            currentGroup.remove(key);
        }
        freq++;
        freqMap.put(key, freq);
        map.put(key, value);
        if (freq < leastFreq) {
            leastFreq = freq;
        }
        LinkedList<Integer> freqList = freqGroup.getOrDefault(freq, new LinkedList<>());
        freqList.addLast(key);
        freqGroup.put(freq, freqList);

    }
}


class LFUCache2 {

    HashMap<Integer, Integer> vals;
    HashMap<Integer, Integer> counts;
    //Use LinkedHashSet to let remove from O(N) to O(1)
    HashMap<Integer, LinkedList<Integer>> lists;
    int cap;
    int min = -1;

    public LFUCache2(int capacity) {
        cap = capacity;
        vals = new HashMap<>();
        counts = new HashMap<>();
        lists = new HashMap<>();
        lists.put(1, new LinkedList<>());
    }

    public int get(Integer key) {
        if (!vals.containsKey(key)) {
            return -1;
        }
        int count = counts.get(key);
        counts.put(key, count + 1);
        lists.get(count).remove(key);
        if (count == min && lists.get(count).size() == 0) {
            min++;
        }
        if (!lists.containsKey(count + 1)) {
            lists.put(count + 1, new LinkedList<>());
        }
        lists.get(count + 1).add(key);
        return vals.get(key);
    }

    public void put(Integer key, int value) {
        if (cap <= 0) {
            return;
        }
        if (vals.containsKey(key)) {
            vals.put(key, value);
            get(key);
            return;
        }
        if (vals.size() >= cap) {
            Integer evit = lists.get(min).iterator().next();
            lists.get(min).remove(evit);
            vals.remove(evit);
        }
        vals.put(key, value);
        counts.put(key, 1);
        min = 1;
        lists.get(1).add(key);
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

class FreqStack {

    Map<Integer, Integer> freq;
    Map<Integer, Stack<Integer>> group;
    int maxfreq;

    public FreqStack() {
        freq = new HashMap();
        group = new HashMap();
        maxfreq = 0;
    }

    public void push(int x) {
        int f = freq.getOrDefault(x, 0) + 1;
        freq.put(x, f);
        if (f > maxfreq) {
            maxfreq = f;
        }
        if (group.containsKey(f)) {
            group.get(f).push(x);
        } else {
            Stack<Integer> s = new Stack<>();
            s.push(x);
            group.put(f, s);
        }
    }

    public int pop() {
        int x = group.get(maxfreq).pop();
        freq.put(x, freq.get(x) - 1);
        if (group.get(maxfreq).size() == 0) {
            group.remove(maxfreq);
            maxfreq--;
        }
        return x;
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

class HitCounter {

    Queue<Integer> q = null;

    /**
     * Initialize your data structure here.
     */
    public HitCounter() {
        q = new LinkedBlockingQueue<>();
    }

    /**
     * Record a hit.
     *
     * @param timestamp - The current timestamp (in seconds granularity).
     */
    public void hit(int timestamp) {
        q.offer(timestamp);
    }

    /**
     * Return the number of hits in the past 5 minutes.
     *
     * @param timestamp - The current timestamp (in seconds granularity).
     */
    public int getHits(int timestamp) {
        while (!q.isEmpty() && timestamp - q.peek() >= 300) {
            q.poll();
        }
        return q.size();
    }
}

class FileSystem {

    Map<String, Integer> pathDict;

    public FileSystem() {
        pathDict = new HashMap<>();
        pathDict.put("/", 0);
    }

    public boolean createPath(String path, int value) {
        if (pathDict.containsKey(path)) {
            return false;
        }
        int lastSlash = path.lastIndexOf("/");
        if (lastSlash != 0) {
            String parentPath = path.substring(0, path.lastIndexOf("/"));
            if (!pathDict.containsKey(parentPath)) {
                return false;
            }
        }
        pathDict.put(path, value);
        return true;
    }

    public int get(String path) {
        return pathDict.containsKey(path) ? pathDict.get(path) : -1;
    }
}
