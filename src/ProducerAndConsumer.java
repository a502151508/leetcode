import java.util.LinkedList;
import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * Created by Sichi Zhang on 2020/4/21.
 */
public class ProducerAndConsumer {


    public static void main(String[] args) {
        BlockingQueue<Integer> q = new ArrayBlockingQueue<>(2);
        Producer producer = new Producer(q);
        Customer c1 = new Customer("c1", q);
        Customer c2 = new Customer("c2", q);
        Customer c3 = new Customer("c3", q);
        producer.start();
        c1.start();
        c2.start();
        c3.start();

    }
}


class ProductWareHouse {

    int size;
    private List<Double> products;

    public ProductWareHouse(int size) {
        this.size = size;
        products = new LinkedList<>();
    }

    public synchronized void produce() {
        while (products.size() == size) {
            try {
                this.wait();
                System.out.println("生产者被唤醒");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        if (products.size() < size) {
            double curProduct = Math.random();
            products.add(curProduct);
            System.out.println("生产了" + curProduct);
            this.notify();
        }
    }

    public synchronized void customer(Customer c) {
        while (products.isEmpty()) {
            try {
                this.wait();
                System.out.println(c.name + "被唤醒");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println(c.name + "消费了" + products.get(0));
        products.remove(0);
        this.notifyAll();
    }
}

class Producer extends Thread {

    BlockingQueue<Integer> q;
    private double pro;

    public Producer(BlockingQueue<Integer> q) {
        this.q = q;
    }

    @Override
    public void run() {
        for (int i = 0; i < 10; i++) {
            try {
                q.put(i);
                System.out.println("生产者生产了" + i);
                Thread.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        }
    }
}


class Customer extends Thread {

    String name;
    BlockingQueue<Integer> q;

    public Customer(String name, BlockingQueue<Integer> q) {
        this.name = name;
        this.q = q;
    }

    @Override
    public void run() {
        while (true) {
            try {
                System.out.println(this.name + "消费了" + q.take());
                Thread.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

    }

}


