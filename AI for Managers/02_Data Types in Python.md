# Python Data Types: A Comprehensive Guide

## Introduction

In Python, data types are the classification of data items that tell the interpreter how the programmer intends to use the data. Think of data types as different containers that hold different kinds of information, each with their own rules and capabilities.

Python has several built-in data types that can be categorized into different groups. Understanding these is fundamental to programming effectively in Python.

## Basic Data Types Categories

Python data types can be broadly categorized as:

- **Numeric Types**: Numbers (integers, floats, complex)
- **Text Type**: Strings
- **Boolean Type**: True/False values
- **Collection Types**: Lists, tuples, sets, dictionaries
- **Special Types**: None type

---

## 1. Numeric Data Types

### Integers (int)

Integers are whole numbers without decimal points. They can be positive, negative, or zero.

```python
# Integer examples
age = 25
temperature = -5
zero = 0
large_number = 1000000

# You can check the type
print(type(age))  # Output: <class 'int'>
```

**Key characteristics:**

- No size limit in Python 3
- Can be written in different bases (binary, octal, hexadecimal)

```python
# Different number bases
binary = 0b1010      # Binary (equals 10 in decimal)
octal = 0o12         # Octal (equals 10 in decimal)
hexadecimal = 0xa    # Hexadecimal (equals 10 in decimal)
```

### Floating Point Numbers (float)

Floats represent numbers with decimal points.

```python
# Float examples
price = 19.99
pi = 3.14159
scientific = 2.5e6  # Scientific notation (2,500,000)
negative_float = -7.5

print(type(price))  # Output: <class 'float'>
```

**Important notes:**

- Floats have limited precision
- Be careful with float comparisons due to rounding errors

```python
# Float precision example
result = 0.1 + 0.2
print(result)  # Output: 0.30000000000000004 (not exactly 0.3!)
```

### Complex Numbers (complex)

Complex numbers have real and imaginary parts.

```python
# Complex number examples
complex_num = 3 + 4j
another_complex = complex(2, -3)  # 2 - 3j

print(complex_num.real)  # Output: 3.0
print(complex_num.imag)  # Output: 4.0
```

---

## 2. String Data Type (str)

Strings represent text data and are sequences of characters enclosed in quotes.

### Creating Strings

```python
# Different ways to create strings
single_quotes = 'Hello World'
double_quotes = "Python Programming"
triple_quotes = """This is a
multi-line string"""

# Empty string
empty = ""
```

### String Operations

```python
# String concatenation
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name
print(full_name)  # Output: John Doe

# String repetition
greeting = "Hello! " * 3
print(greeting)  # Output: Hello! Hello! Hello! 

# String length
message = "Welcome to Python"
print(len(message))  # Output: 17
```

### String Indexing and Slicing

```python
text = "Python Programming"

# Indexing (starts from 0)
print(text[0])    # Output: P
print(text[-1])   # Output: g (last character)

# Slicing
print(text[0:6])  # Output: Python
print(text[7:])   # Output: Programming
print(text[:6])   # Output: Python
```

### Useful String Methods

```python
sample = "  Hello World  "

# Case methods
print(sample.upper())        # Output: "  HELLO WORLD  "
print(sample.lower())        # Output: "  hello world  "
print(sample.title())        # Output: "  Hello World  "

# Whitespace methods
print(sample.strip())        # Output: "Hello World"
print(sample.lstrip())       # Output: "Hello World  "

# Search and replace
print(sample.replace("World", "Python"))  # Output: "  Hello Python  "
print(sample.find("World"))               # Output: 8
```

### String Formatting

```python
# F-strings (Python 3.6+) - Recommended
name = "Alice"
age = 30
message = f"My name is {name} and I am {age} years old."

# .format() method
message2 = "My name is {} and I am {} years old.".format(name, age)

# % formatting (older method)
message3 = "My name is %s and I am %d years old." % (name, age)
```

---

## 3. Boolean Data Type (bool)

Booleans represent truth values: True or False.

```python
# Boolean values
is_student = True
is_adult = False

# Boolean operations
print(True and False)  # Output: False
print(True or False)   # Output: True
print(not True)        # Output: False

# Comparison operations return booleans
print(5 > 3)          # Output: True
print(10 == 5)        # Output: False
```

### Truthiness in Python

Many values can be evaluated as True or False:

```python
# Values that are considered False (Falsy)
print(bool(0))        # False
print(bool(""))       # False (empty string)
print(bool([]))       # False (empty list)
print(bool(None))     # False

# Values that are considered True (Truthy)
print(bool(1))        # True
print(bool("hello"))  # True
print(bool([1, 2]))   # True
```

---

## 4. Collection Data Types

### Lists (list)

Lists are ordered, mutable collections that can store multiple items.

```python
# Creating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = ["hello", 42, True, 3.14]
empty_list = []

# List operations
fruits.append("grape")          # Add item
fruits.insert(1, "mango")      # Insert at position
removed = fruits.pop()         # Remove and return last item
fruits.remove("banana")        # Remove specific item

# List indexing and slicing (same as strings)
print(fruits[0])               # First item
print(fruits[-1])              # Last item
print(fruits[1:3])             # Slice
```

### Tuples (tuple)

Tuples are ordered, immutable collections.

```python
# Creating tuples
coordinates = (10, 20)
colors = ("red", "green", "blue")
single_item = (42,)           # Note the comma for single item
empty_tuple = ()

# Tuple operations (limited due to immutability)
x, y = coordinates            # Tuple unpacking
print(colors.count("red"))    # Count occurrences
print(colors.index("green"))  # Find index
```

### Sets (set)

Sets are unordered collections of unique items.

```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
letters = set("hello")        # {'h', 'e', 'l', 'o'}
empty_set = set()             # Note: {} creates a dict, not a set

# Set operations
unique_numbers.add(6)         # Add item
unique_numbers.remove(1)      # Remove item
unique_numbers.discard(10)    # Remove if exists (no error if not)

# Set mathematical operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print(set1.union(set2))       # {1, 2, 3, 4, 5, 6}
print(set1.intersection(set2)) # {3, 4}
```

### Dictionaries (dict)

Dictionaries store key-value pairs.

```python
# Creating dictionaries
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science"
}

# Dictionary operations
print(student["name"])        # Access value
student["gpa"] = 3.8         # Add new key-value pair
student["age"] = 21          # Update existing value

# Dictionary methods
print(student.keys())        # Get all keys
print(student.values())      # Get all values
print(student.items())       # Get key-value pairs

# Safe access
print(student.get("height", "Not specified"))  # Returns default if key doesn't exist
```

---

## 5. None Type

None represents the absence of a value or null value.

```python
# None examples
result = None
name = None

# Common use cases
def greet(name=None):
    if name is None:
        print("Hello, stranger!")
    else:
        print(f"Hello, {name}!")

# Checking for None
if result is None:
    print("No result available")
```

---

## Type Conversion and Checking

### Checking Types

```python
value = 42
print(type(value))           # <class 'int'>
print(isinstance(value, int)) # True
print(isinstance(value, str)) # False
```

### Type Conversion

```python
# Converting between types
number_str = "123"
number_int = int(number_str)     # String to integer
number_float = float(number_str)  # String to float

# Converting numbers to strings
age = 25
age_str = str(age)

# Converting to boolean
print(bool(1))      # True
print(bool(0))      # False
print(bool(""))     # False
print(bool("hi"))   # True

# Converting lists and strings
text = "hello"
char_list = list(text)          # ['h', 'e', 'l', 'l', 'o']
back_to_string = "".join(char_list)  # "hello"
```

---

## Practical Examples

### Example 1: Working with Mixed Data Types

```python
# Student information system
students = [
    {"name": "Alice", "age": 20, "grades": [85, 92, 78]},
    {"name": "Bob", "age": 19, "grades": [90, 88, 95]},
    {"name": "Charlie", "age": 21, "grades": [76, 82, 89]}
]

# Process the data
for student in students:
    name = student["name"]
    average_grade = sum(student["grades"]) / len(student["grades"])
    print(f"{name}'s average grade: {average_grade:.2f}")
```

### Example 2: Data Type Validation

```python
def validate_user_input(data):
    """Validate different types of user input"""
    errors = []

    # Check name (should be string)
    if not isinstance(data.get("name"), str) or not data.get("name").strip():
        errors.append("Name must be a non-empty string")

    # Check age (should be positive integer)
    age = data.get("age")
    if not isinstance(age, int) or age < 0:
        errors.append("Age must be a positive integer")

    # Check email (should be string with @ symbol)
    email = data.get("email")
    if not isinstance(email, str) or "@" not in email:
        errors.append("Email must be a valid email address")

    return errors

# Test the function
user_data = {"name": "John", "age": 25, "email": "john@example.com"}
validation_errors = validate_user_input(user_data)
if validation_errors:
    print("Validation errors:", validation_errors)
else:
    print("Data is valid!")
```

---

## Key Takeaways

1. **Choose the right data type**: Use integers for whole numbers, floats for decimals, strings for text, booleans for True/False values.

2. **Mutability matters**: Lists and dictionaries can be changed after creation, but tuples and strings cannot.

3. **Type conversion**: Python can convert between compatible types, but be careful about data loss.

4. **Collections serve different purposes**:
   
   - Lists for ordered, changeable data
   - Tuples for ordered, unchangeable data
   - Sets for unique items
   - Dictionaries for key-value relationships

5. **Always validate data types**: Use `isinstance()` to check types and handle user input safely.

Understanding these data types and their characteristics is crucial for writing effective Python programs and working with data in data science applications.
