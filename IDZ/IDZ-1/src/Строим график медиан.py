import csv
import statistics
import matplotlib.pyplot as plt

def plot_medians(filename: str):
    block_size = 100_000        # 10^5 записей
    blocks = [[] for _ in range(10)]  # 10 кортежей

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)

        index = 0
        for row in reader:
            value = float(row['func4'])
            block_index = index // block_size
            if block_index < 10:
                blocks[block_index].append(value)
            index += 1

    # вычисляем медианы каждого блока
    medians = [statistics.median(block) for block in blocks]

    # строим график
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), medians, marker='o')
    plt.xlabel("Номер кортежа (1–10)")
    plt.ylabel("Медиана значений func4")
    plt.title("График медиан func4 по кортежам из 10^5 записей")
    plt.grid(True)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("График медиан func4.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_medians('test.csv')
    print("График медиан func4 успешно построен.")