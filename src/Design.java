import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
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
    private Map<Integer, LinkedHashSet<Integer>> freqGroup;
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
            LinkedHashSet<Integer> currentGroup = freqGroup.get(freq);
            currentGroup.remove(key);
            freq++;
            freqMap.put(key, freq);
            LinkedHashSet<Integer> freqList = freqGroup.getOrDefault(freq, new LinkedHashSet<>());
            freqList.add(key);
            freqGroup.put(freq, freqList);
        }
        return res;
    }

    public void put(Integer key, int value) {
        if (capacity <= 0) {
            return;
        }
        if (map.size() == capacity && !map.containsKey(key)) {
            LinkedHashSet<Integer> leastFreqGroup = freqGroup.get(leastFreq);
            while (leastFreqGroup == null || leastFreqGroup.size() == 0) {
                leastFreq++;
                leastFreqGroup = freqGroup.getOrDefault(leastFreq, null);
            }
            int removedKey = leastFreqGroup.iterator().next();
            leastFreqGroup.remove(removedKey);
            freqMap.remove(removedKey);
            map.remove(removedKey);
        }

        int freq = freqMap.getOrDefault(key, 0);
        if (freq > 0) {
            LinkedHashSet<Integer> currentGroup = freqGroup.get(freq);
            currentGroup.remove(key);
        }
        freq++;
        freqMap.put(key, freq);
        map.put(key, value);
        if (freq < leastFreq) {
            leastFreq = freq;
        }
        LinkedHashSet<Integer> freqList = freqGroup.getOrDefault(freq, new LinkedHashSet<>());
        freqList.add(key);
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


/*
895. Maximum Frequency Stack
one map to store element's count,
another map to store a stack for all elements with same frequency. stack can keep the order.
Time O(1)
space O(N)
 */
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


/*
LC  380. Insert Delete GetRandom O(1)
using hashmap to record data -> index in arrayList
when add, add it to the end of the list, and add val -> index to the hashmap.
when delete, swap the delete one with the last one in list. delete the last one, remove the val from hashmap
when getRandom, just get a random number in list.size,and return that element, array list can do that in O(1)
Time O(1)
sapce O(N)to store data
 */
class RandomizedSet {

    Map<Integer, Integer> dict;
    List<Integer> list;
    Random rand;

    /**
     * Initialize your data structure here.
     */
    public RandomizedSet() {
        dict = new HashMap();
        list = new ArrayList();
        rand = new Random();
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
        if (list.size() > 0) {
            return list.get(rand.nextInt(list.size()));
        } else {
            return -1;
        }
    }
}


/*
lc 381. Insert Delete GetRandom O(1) - Duplicates allowed
time O(1)
space O(N)
 */
class RandomizedCollection {

    Map<Integer, Set<Integer>> map;
    List<Integer> data;
    Random rand;

    /**
     * Initialize your data structure here.
     */
    public RandomizedCollection() {
        rand = new Random();
        map = new HashMap<>();
        data = new ArrayList<>();
    }

    /**
     * Inserts a value to the collection. Returns true if the collection did not already contain the
     * specified element.
     */
    public boolean insert(int val) {
        //linkedList is good for many insert and delete, which can be Done in O(1)
        Set<Integer> indexes = map.getOrDefault(val, new HashSet<>());
        indexes.add(data.size());
        data.add(val);
        map.put(val, indexes);
        return indexes.size() == 1;
    }

    /**
     * Removes a value from the collection. Returns true if the collection contained the specified
     * element.
     */
    public boolean remove(int val) {
        if (map.containsKey(val)) {
            Set<Integer> indexes = map.get(val);
            int index = indexes.iterator().next();
            //delete from index dict
            indexes.remove(index);
            if (indexes.size() == 0) {
                map.remove(val);
            }
            //delete from data list
            int lastElement = data.get(data.size() - 1);
            //if lastElement is not been deleted.
            if (map.containsKey(lastElement)) {
                Set<Integer> lastElementIndexes = map.get(lastElement);
                lastElementIndexes.remove(data.size() - 1);
                lastElementIndexes.add(index);
            }
            data.set(index, lastElement);
            data.remove(data.size() - 1);
            return true;
        }
        return false;

    }

    /**
     * Get a random element from the collection.
     */
    public int getRandom() {
        return data.get(rand.nextInt(data.size()));
    }
}


/*
Fisher–Yates
一、算法流程：
需要随机置乱的n个元素的数组a：
for i 从n-1到1

j <—随机整数(0 =< j <= i)

交换a[i]和a[j]

 end
 */
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

class TicTacToe {

    private int[] rows;
    private int[] cols;
    private int diagonal;
    private int antiDiagonal;

    /**
     * Initialize your data structure here.
     */
    public TicTacToe(int n) {
        rows = new int[n];
        cols = new int[n];
    }

    /**
     * Player {player} makes a move at ({row}, {col}).
     *
     * @param row The row of the board.
     * @param col The column of the board.
     * @param player The player, can be either 1 or 2.
     * @return The current winning condition, can be either: 0: No one wins. 1: Player 1 wins. 2:
     * Player 2 wins.
     */
    public int move(int row, int col, int player) {
        int toAdd = player == 1 ? 1 : -1;

        rows[row] += toAdd;
        cols[col] += toAdd;
        if (row == col) {
            diagonal += toAdd;
        }

        if (col == (cols.length - row - 1)) {
            antiDiagonal += toAdd;
        }

        int size = rows.length;
        if (Math.abs(rows[row]) == size ||
            Math.abs(cols[col]) == size ||
            Math.abs(diagonal) == size ||
            Math.abs(antiDiagonal) == size) {
            return player;
        }

        return 0;
    }
}

/*
lc 706 design a hashmap
 */

class MyHashMap {

    Node[] table;

    private class Node {

        Node next;
        int key;
        int val;

        public Node(int key, int val) {
            this.key = key;
            this.val = val;
        }
    }

    /**
     * Initialize your data structure here.
     */
    public MyHashMap() {
        table = new Node[1024];
    }

    /**
     * value will always be non-negative.
     */
    public void put(int key, int value) {
        int index = findIndex(Integer.hashCode(key));
        if (table[index] == null) {
            table[index] = new Node(key, value);
        } else {
            Node cur = table[index];
            Node prev = null;
            while (cur != null) {
                //find an old one ,edit it.
                if (cur.key == key) {
                    cur.val = value;
                    return;
                }
                prev = cur;
                cur = cur.next;
            }
            prev.next = new Node(key, value);
        }
    }

    /**
     * Returns the value to which the specified key is mapped, or -1 if this map contains no mapping
     * for the key
     */
    public int get(int key) {
        Node e = findNode(key);
        return e == null ? -1 : e.val;
    }

    /**
     * Removes the mapping of the specified value key if this map contains a mapping for the key
     */
    public void remove(int key) {
        Node prev = null;
        if (table[findIndex(Integer.hashCode(key))] == null) {
            return;
        } else {
            Node cur = table[findIndex(Integer.hashCode(key))];
            if (cur.key == key) {
                table[findIndex(Integer.hashCode(key))] = cur.next;
                return;
            }
            while (cur != null && cur.key != key) {
                prev = cur;
                cur = cur.next;
            }
            if (cur == null) {
                return;
            } else {
                prev.next = cur.next;
                cur = null;
            }
        }
    }

    private Node findNode(int key) {
        if (table[findIndex(Integer.hashCode(key))] == null) {
            return null;
        } else {
            Node cur = table[findIndex(Integer.hashCode(key))];
            while (cur != null && cur.key != key) {
                cur = cur.next;
            }
            if (cur == null) {
                return null;
            } else {
                return cur;
            }
        }
    }

    private int findIndex(int hash) {
        return (table.length - 1) & hash;
    }
}

class MedianFinder {

    PriorityQueue<Integer> rightPartMinHeap;
    PriorityQueue<Integer> leftPartMaxHeap;

    /**
     * initialize your data structure here.
     */
    public MedianFinder() {
        rightPartMinHeap = new PriorityQueue<>(Comparator.comparingInt(o -> o));
        leftPartMaxHeap = new PriorityQueue<>(Comparator.comparingInt(o -> -o));
    }

    public void addNum(int num) {
        //init
        if (leftPartMaxHeap.isEmpty()) {
            leftPartMaxHeap.add(num);
        } else if (rightPartMinHeap.isEmpty()) {
            if (num > leftPartMaxHeap.peek()) {
                rightPartMinHeap.add(num);
            } else {
                rightPartMinHeap.add(leftPartMaxHeap.poll());
                leftPartMaxHeap.add(num);
            }
        }
        //when two heap both have elements.
        else {
            int leftMax = leftPartMaxHeap.peek();
            int rightMin = rightPartMinHeap.peek();
            if (leftPartMaxHeap.size() == rightPartMinHeap.size()) {
                if (num < rightMin) {
                    leftPartMaxHeap.add(num);
                } else {
                    leftPartMaxHeap.add(rightPartMinHeap.poll());
                    rightPartMinHeap.add(num);
                }
            }
            //this means left size is one more large than right, we should place one to right.
            else {
                if (num > leftMax) {
                    rightPartMinHeap.add(num);
                } else {
                    rightPartMinHeap.add(leftPartMaxHeap.poll());
                    leftPartMaxHeap.add(num);
                }
            }
        }
    }

    public double findMedian() {
        if (leftPartMaxHeap.size() == rightPartMinHeap.size()) {
            return (leftPartMaxHeap.peek() + rightPartMinHeap.peek()) / 2.0;
        } else {
            return leftPartMaxHeap.peek();
        }
    }
}