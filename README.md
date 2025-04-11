# CPP-Compiler

## Project Overview

The **CPP-Compiler** is a basic lexical analyzer implemented in C++. It reads a source code file (`input.cc`), tokenizes it, and outputs the tokens and their lexemes to the console and a file (`output.txt`). This project demonstrates the fundamental concepts of lexical analysis, including recognizing keywords, identifiers, constants, operators, and other language constructs.

## Features

- Tokenizes source code into meaningful components (tokens and lexemes).
- Supports:
  - Keywords (`INT`, `CHAR`, `IF`, `ELSE`, etc.).
  - Identifiers.
  - Numeric constants.
  - String and literal constants.
  - Operators (arithmetic, relational, assignment).
  - Separators (parentheses, braces, etc.).
  - Comments.
- Outputs tokens and lexemes to both the console and a file (`output.txt`).
- Reports errors with line and column numbers for invalid tokens.

---

## How to Use

### Prerequisites

- A C++ compiler (e.g., GCC, Clang, or MSVC).
- A source code file named input.cc in the same directory as the program.

### Compilation

1. Open a terminal or command prompt.
2. Navigate to the directory containing the Source.cpp file.
3. Compile the program using your C++ compiler. For example:
   ```bash
   g++ Source.cpp -o CPP-Compiler
   ```

### Running the Program

1. Ensure the input.cc file contains the source code you want to analyze.
2. Run the compiled program:
   ```bash
   ./CPP-Compiler   # For Linux/Mac
   CPP-Compiler.exe # For Windows
   ```

### Output

- The program will display the tokens and their lexemes in the console.
- The same output will be saved to a file named output.txt in the same directory.

---

## Example

### Input (`input.cc`)

```cpp
int x = 10;
char y = 'a';
if (x > 5) {
    print("Hello, World!");
}
```

### Console Output

```
('INT', '^')
('ID', 'x')
('=', '^')
('NUM', '10')
(';', '^')
('CHAR', '^')
('ID', 'y')
('=', '^')
('LIT', 'a')
(';', '^')
('CS', 'IF')
('(', '^')
('ID', 'x')
('RO', '>')
('NUM', '5')
(')', '^')
('{', '^')
('PRINT', '^')
('STR', 'Hello, World!')
(')', '^')
(';', '^')
('}', '^')
```

### File Output (`output.txt`)

The same as the console output.

---

## Error Handling

If the program encounters an invalid token, it will display an error message with the token number, line number, and column number. For example:

```
Error at Token Number 15, Located at Line 3, Column 7
```

---

## Project Structure

- **Source.cpp**: The main source code for the lexical analyzer.
- **input.cc**: The input file containing the source code to be analyzed.
- **output.txt**: The file where the tokens and lexemes are saved.

---

## Notes

- Ensure the input.cc file is present in the same directory as the program before running it.
- The program assumes a maximum input size of 10,240 characters. Adjust the `data` array size in `main()` if needed.

---

## License

This project is open-source and available for educational purposes. Feel free to modify and use it as needed.
