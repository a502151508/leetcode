import java.util.LinkedList;
import java.util.*;

/**
 * Created by Sichi Zhang on 2020/4/21.
 */
public class ProducerAndConsumer {


    public static void main(String[] args) {
        ProductWareHouse p = new ProductWareHouse(10);
        Producer producer = new Producer(p);
        Customer c1 = new Customer("c1", p);
        Customer c2 = new Customer("c2", p);
        Customer c3 = new Customer("c3", p);
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
        if (products.size() == size) {
            try {
                this.wait();
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

    ProductWareHouse p;

    public Producer(ProductWareHouse p) {
        this.p = p;
    }

    @Override
    public void run() {
        for (int i = 0; i < 100; i++) {
            p.produce();
            try {
                Thread.sleep(1000);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}


class Customer extends Thread {

    String name;
    ProductWareHouse p;

    public Customer(String name, ProductWareHouse p) {
        this.name = name;
        this.p = p;
    }

    @Override
    public void run() {
        while (true) {
            p.customer(this);
            try {
                Thread.sleep(1000);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

}


