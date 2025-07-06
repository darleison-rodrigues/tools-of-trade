# sample_code.py
def calculate_fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a + b

if __name__ == '__main__':
    calculate_fibonacci(10)

