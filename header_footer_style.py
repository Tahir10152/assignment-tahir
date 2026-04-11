import random

def header_style(command, min_count=50, max_count=55):

    random_number = random.randint(min_count, max_count)

    print("\n" + "=" * random_number)
    print(f"  {command}")
    print("=" * random_number)

