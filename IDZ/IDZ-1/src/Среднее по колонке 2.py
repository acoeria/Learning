import csv

def read_and_avg(filename: str):
    total = 0.0
    count = 0

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total += float(row['func1'])
            count += 1

    return total / count


if __name__ == '__main__':
    avg = read_and_avg('test.csv')
    print("Среднее по колонке func1:", avg)