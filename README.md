# CPP-Compiler

## Project Overview

The **CPP-Compiler** is a simple compiler implementation written in C++. It performs lexical analysis, parsing, intermediate code generation (Three Address Code - TAC), and machine code generation. The project also includes a virtual machine (VM) to execute the generated machine code.

This project demonstrates the fundamental concepts of compiler design, including tokenization, parsing, symbol table management, and code execution.

---

## Features

1. **Lexical Analysis**:

   - Tokenizes the input source code into meaningful components (tokens and lexemes).
   - Supports keywords, identifiers, numeric constants, string literals, operators, and separators.

2. **Parsing**:

   - Implements a recursive descent parser to validate the syntax of the input code.
   - Generates a symbol table and TAC (Three Address Code).

3. **Machine Code Generation**:

   - Converts TAC into machine code instructions.
   - Stores machine code in memory for execution.

4. **Virtual Machine (VM)**:
   - Executes the generated machine code.
   - Supports input/output operations, arithmetic, and control flow.

---

## Project Structure

- **Lexer**: Handles lexical analysis and tokenization.
- **Parser**: Performs syntax analysis and generates TAC.
- **MachineCodeGenerator**: Converts TAC into machine code.
- **VM**: Executes the machine code.

---

## How to Use

### Prerequisites

- A C++ compiler (e.g., GCC, Clang, or MSVC).
- A source code file named input.cc containing the code to be compiled.

### Compilation

1. Open a terminal or command prompt.
2. Navigate to the directory containing the Source.cpp file.
3. Compile the program using your C++ compiler. For example:
   ```bash
   g++ Source.cpp -o CPP-Compiler
   ```

### Running the Program

1. Ensure the input.cc file contains the source code you want to compile.
2. Run the compiled program:
   ```bash
   ./CPP-Compiler   # For Linux/Mac
   CPP-Compiler.exe # For Windows
   ```

### Output

- **Tokens and Lexemes**: Saved in output.txt.
- **Symbol Table**: Saved in symbolTable.txt.
- **Three Address Code (TAC)**: Saved in TAC.txt.
- **Machine Code**: Saved in MCG.txt.
- **Execution Output**: Displayed in the console.

---

## Example

### Input (`input.cc`)

```cpp
int: num;
char: my_char;
my_char = 'd';
print("my char contains: ");
println(my_char);
```

### Output

#### Tokens (`output.txt`)

```
('INT', '^')
('ID', 'num')
(';', '^')
('CHAR', '^')
('ID', 'my_char')
(';', '^')
('ID', 'my_char')
('=', '^')
('LIT', 'd')
(';', '^')
('PRINT', '^')
('STR', 'my char contains: ')
(';', '^')
('PRINTLN', '^')
('ID', 'my_char')
(';', '^')
```

#### Symbol Table (`symbolTable.txt`)

```
('INT', 'num, 0')
('CHAR', 'my_char, 4')
```

#### Three Address Code (`TAC.txt`)

```
my_char = 'd'
OUT "my char contains: "
OUT my_char \n
```

#### Machine Code (`MCG.txt`)

```
18 100 4
15 104 0
14 4 1
```

#### Execution Output

```
my char contains:
d
```

---

## Key Components

### Lexer

- Tokenizes the input source code.
- Handles keywords, identifiers, constants, operators, and separators.

### Parser

- Validates the syntax of the input code.
- Generates a symbol table and TAC.

### Machine Code Generator

- Converts TAC into machine code instructions.
- Stores machine code in memory.

### Virtual Machine (VM)

- Executes the machine code.
- Supports arithmetic, input/output, and control flow.

---

## Notes

- Ensure the input.cc file is present in the same directory as the program before running it.
- The program assumes a maximum input size of 10,240 characters. Adjust the `str` array size in the `lex` function if needed.

---

## Limitations

- The compiler currently supports a limited subset of C-like syntax.
- Error handling is basic and may need improvement for larger programs.

---

## License

This project is open-source and available for educational purposes. Feel free to modify and use it as needed.
