import csv
import matplotlib.pyplot as plt

def plot_columns(filename: str):
    times = []
    col3 = []  # func2
    col4 = []  # func3

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            col3.append(float(row['func2']))
            col4.append(float(row['func3']))

    # Строим графики
    plt.figure(figsize=(10, 5))
    plt.plot(times, col3, label='func2 (колонка 3)')
    plt.plot(times, col4, label='func3 (колонка 4)')

    plt.xlabel('time (колонка 1)')
    plt.ylabel('значения функций')
    plt.title('График func2 и func3 в зависимости от time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("График значений для колонок 3 и 4.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_columns('test.csv')
    print("График успешно сохранён ")
