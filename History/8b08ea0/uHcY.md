
# fingerprint
- https://en.wikipedia.org/wiki/Fingerprint_(computing)
- In computer science, a fingerprinting algorithm is a procedure that maps an arbitrarily large data item (such as a computer file) to a much shorter bit string, its fingerprint, that uniquely identifies the original data for all practical purposes[1] just as human fingerprints uniquely identify people for practical purposes. This fingerprint may be used for data deduplication purposes. This is also referred to as file fingerprinting, data fingerprinting, or structured data fingerprinting.
- Mainstream cryptographic grade hash functions generally can serve as high-quality fingerprint functions, are subject to intense scrutiny from cryptanalysts, and have the advantage that they are believed to be safe against malicious attacks.
- A drawback of cryptographic hash algorithms such as MD5 and SHA is that they take considerably longer to execute than Rabin's fingerprint algorithm. They also lack proven guarantees on the collision probability. Some of these algorithms, notably MD5, are no longer recommended for secure fingerprinting. They are still useful for error checking, where purposeful data tampering isn't a primary concern.
- Cryptographic hash function != fingerprint function!
- one-way function: a function for which it is practically infeasible to invert or reverse the computation. Ideally, the only way to find a message that produces a given hash is to attempt a brute-force search of possible inputs to see if they produce a match, or use a rainbow table of matched hashes. Cryptographic hash functions are a basic tool of modern cryptography.
- 



# memory
## page
- page/memory page/virtual page: is a fixed-length contiguous block of virtual memory, described by a single entry in the page table. It is the smallest unit of data for memory management in a virtual memory operating system.
- page frame: is the smallest fixed-length contiguous block of physical memory into which memory pages are mapped by the operating system.
- paging/swapping: A transfer of pages between main memory and an auxiliary store, such as a hard disk drive, is referred to as paging or swapping


# new stuff for malware
- https://nakedsecurity.sophos.com/2012/07/31/server-side-polymorphism-malware/#:~:text=Server%2Dside%20polymorphism%20is%20a,no%20sample%20looks%20the%20same.
- `Server-side polymorphism` is a technique used by malware distributors in an attempt to evade detection by anti-virus software. Regular polymorphic (literally “many shapes”) malwa4re is malicious code which changes its appearance through obfuscation and encryption, ensuring that no sample looks the same.
- http://anti-virus-rants.blogspot.com/2007/08/what-is-server-side-polymorphism.html
- server-side polymorphism is a type of polymorphism where the polymorphic engine (the transformation function responsible for producing the malware's many forms) doesn't reside within the malware itself
- fingerprint vs digital signature: https://stackoverflow.com/questions/10546837/public-key-fingerprint-vs-digital-signature#:~:text=The%20fingerprint%20is%20the%20hash,encrypted%20hash%20of%20the%20message.
    - The fingerprint is the hash of a key. A digital signature is tied to some message, and is typically a one-way encrypted hash of the message.
- `Digital signature`: A digital signature is a mathematical scheme for verifying the authenticity of digital messages or documents. A valid digital signature, where the prerequisites are satisfied, gives a recipient very high confidence that the message was created by a known sender, and that the message was not altered in transit.


- `Decryptors`: decryption tool

# research directions for malware
- new benign files that go unseen earlier may occasionally be falsely detected. We take this into account and implement a flexible design of a model that allows us to fix false-positives on the fly, without completely retraining the model. Examples of this are implemented in our pre- and post-execution models, which are described in the following sections.
- Algorithms must allow us to quickly adapt
them to malware writers’ counteractions
After applying machine learning to malware detection, we have to face the fact that our data distribution isn’t fixed:
    - Active adversaries (malware writers) constantly work on avoiding detections and releasing new versions of malware files that differ significantly from those that have been seen during the training phase.
    - Thousands of software companies produce new types of benign executables that are significantly different from previously known types. The data on these types was lacking in the training set, but the model, nevertheless, needs to recognize them as benign.
- The aforementioned properties of real world malware detection make straightforward application of machine learning techniques a challenging task. Kaspersky has almost a decade’s worth of experience when it comes to utilizing machine learning methods in information security applications.
- At the dawn of the antivirus industry, malware detection on computers was based on heuristic features that identified particular malware files by:
    - code fragments
    - hashes of code fragments or the whole file 
    - file properties
    - and combinations of these features.
- LSH: Regular cryptographic hashes of two almost identical files differ as much as hashes of two very different files. There is no connection between the similarity of files and their hashes. However, LSHs of almost identical files map to the same binary bucket – their LSHs are very similar – with high probability. LSHs of two different files differ substantially.
- Some file features important for detection require larger computational resources
for their calculation. Those features are called “heavy”. To avoid their calculation for
all scanned files, we introduced a preliminary stage called a pre-detect. A pre-detect occurs when a file is analyzed with ‘lightweight’ features and is extracted without substantial load on the system. In many cases, a pre-detect provides us with enough information to know if a file is benign and ends the file scan. Sometimes it even detects a file as malware. If the first stage was not sufficient, the file goes to the second stage of analysis, when ‘heavy’ features are extracted for precise detection.





# DLL
- which are libraries that provide APIs
- 




# things to learn
- what is x86 instructions
- every file is a binary file. But when opened, each file runs or is presented to the user differently based on the file’s extension or data format. A file’s every byte can be visualized in its hex form
- we can open any file in its binary form


# new
- `file extension` can be changed arbitrarily, but `file format` can't be changed!
- to learn the true `file format`, we need to check the first two bytes!
    - suppose you receive a file without extension, how do to know what extension u should give it?
    - check the first several bytes for file format
    - see `file <file_path>` in terminal to learn the file format
- program -> go into a process after execution
- Each process has a name, which is the name of the program from which it was created. The process name is not unique in the list of processes. Two or more processes can have the same name without conflict. Similarly, multiple processes from the same program file can be created, meaning multiple processes may not only have the same name but the same program executable path.
- PID: To uniquely identify a process, each process is given a unique ID called the process ID, or PID, by the OS. PID is a randomly assigned ID/number, and it changes each time the program is executed even on the very same system.
- MZ magic bytes = windows executable. 
    - MZ refers to Mark Zbikowski
- endian: a way to store data in computer systems.
- addressable: Virtual memory, just like physical memory or the RAM, is addressable (i.e., every byte in memory of the process has an address).



## process
- 



# compile file header (file signature)
- `MZ` = (4d 5a): windows PE files
- 


# dynamic vs static program analysis
- https://blog.quarkslab.com/exploring-execution-trace-analysis.html
- Dynamic program analysis consists in examining a program's behavior by processing information captured at execution time. Compared to static analysis, dynamic analysis unveils the actual behavior of a program and provides direct access to its execution flow and data. This is why dynamic analysis can be a powerful weapon when it comes to software reverse engineering (although it doesn't come without drawbacks).


# compile
- remember that the executable file generated from the compiler is processor-dependent!!
- When programmers create software programs, they first write the program in source code, which is written in a specific programming language, such as C or Java. These source code files are saved in a text-based, human-readable format, which can be opened and edited by programmers. However, the source code cannot be run directly by the computer. In order for the code to be recognized by the computer's CPU, it must be converted from source code (a high-level language) into machine code (a low-level language). This process is referred to as "compiling" the code.
- Most software development programs include a compiler, which translates source code files into machine code or object code. Since this code can be executed directly by the computer's processor, the resulting application is often referred to as an executable file. Windows executable files have a .EXE file extension, while Mac OS X programs have an .APP extension, which is often hidden.
- A compiler is a software program that compiles program source code files into an executable program. It is included as part of the integrated development environment IDE with most programming software packages.
- The compiler takes source code files that are written in a high-level language, such as C, BASIC, or Java, and compiles the code into a low-level language, such as machine code or assembly code. This code is created for a specific processor type, such as an Intel Pentium or PowerPC. The program can then be recognized by the processor and run from the operating system.
- After a compiler compiles source code files into a program, the program cannot be modified. Therefore, any changes must be made in the source code and the program must be recompiled. Fortunately, most modern compilers can detect what changes were made and only need to recompile the modified files, which saves programmers a lot of time. This can help reduce programmers' 100 hour work weeks before project deadlines to around 90 or so.

# container
- https://techterms.com/definition/container
- 


# assembler
- https://techterms.com/definition/assembler
- An assembler is a program that converts assembly language into machine code. It takes the basic commands and operations from assembly code and converts them into binary code that can be recognized by a specific type of processor.
- Assemblers are similar to compilers in that they produce executable code. However, assemblers are more simplistic since they only convert low-level code (assembly language) to machine code. **Since each assembly language is designed for a specific processor, assembling a program is performed using a simple one-to-one mapping from assembly code to machine code.** Compilers, on the other hand, must convert generic high-level source code into machine code for a specific processor.
- Most programs are written in high-level programming languages and are compiled directly to machine code using a compiler. However, in some cases, assembly code may be used to customize functions and ensure they perform in a specific way. Therefore, IDEs often include assemblers so they can build programs from both high and low-level languages.


- binary file: usually it means executable file, since all executables are complied and so are in binary.


# software for analysis
- Hiew: http://www.hiew.ru/ 
    - viewing the binary code of a PE
- Interactive Disassembler (IDA): https://hex-rays.com/ida-pro/
    - 

# compile vs interpret
- for compile, it's faster at run time since "run time" doesn't include the time for compilation. 
- the advantage of compiled language is that once you've compiled the code, if we need to run the same compiled code again, we don't need to compile it again.

- The following line is a disassembled x86 code: `68 73 9D 00 01       PUSH 0x01009D73`
    - 68 is the opcode. With the following four bytes it represents PUSH instruction of x86 Assembly language. PUSH instruction pushes 4 bytes (32 bits) length data to stack. The word PUSH is just a mnemonic that represents opcode 68. Each of bytes 68, 73, 9D, 00, 01 is machine code.
    - machine codes are for real machines (CPUs) but byte codes are pseudo machine codes for virtual machines.
    - When you write a java code, java compiler compiles your code and generates byte codes. (A .class file) and you can execute the same code at any platform without changing.
    - 


# python running process
- the standard implementations of Python today compile (i.e., translate) source code statements to an intermediate format known as byte code and then interpret the byte code. **Byte code provides portability, as it is a platform-independent format**. However, because Python is not normally compiled all the way down to binary machine code (e.g., instructions for an Intel chip), some programs will run more slowly in Python than in a fully compiled language like C. The PyPy system discussed in the next chapter can achieve a 10X to 100X speedup on some code by compiling further as your program runs, but it’s a separate, alternative implementation.

# dynamic vs static types:
- `strongly type and weakly type`: in `"3"+5`, raise a type error in strongly typed languages, such as Python and Go, because they don't allow for "type coercion": the ability for a value to change type implicitly in certain contexts (e.g. merging two types using +). Weakly typed languages, such as JavaScript, won't throw a type error (result: '35').
- `dynamical type`: Python keeps track of the kinds of objects your program uses when it runs; it doesn’t require complicated type and size declarations in your code. In fact, as you’ll see in Chapter 6, there is no such thing as a type or variable declaration anywhere in Python. Because Python code does not constrain data types, it is also usually automatically applicable to a whole range of objects.
- Static: Types checked before run-time
- Dynamic: Types checked on the fly, during execution
```python
def foo(a):
    if a > 0:
        print 'Hi'
    else: 
        print "3" + 5 # no error if dynamic type, since this isn't executed
foo(2)
```
- Because Python is both interpreted and dynamically typed, it only translates your location and type-checks code it’s executing on. The else block never executes, so "3" + 5 is never even looked at!
- What if it was statically typed?
    - A type error would be thrown before the code is even run. It still performs type-checking before run-time even though it is interpreted.
- What if it was compiled?
    - The else block would be translated/looked at before run-time, but because it's dynamically typed it wouldn't throw an error! Dynamically typed languages don't check types until execution, and that line never executes.
- A compiled language will have better performance at run-time if it’s statically typed because the knowledge of types allows for machine code optimization.
- Statically typed languages have better performance at run-time intrinsically due to not needing to check types dynamically while executing (it checks before running).
- Similarly, compiled languages are faster at run time as the code has already been translated instead of needing to “interpret”/translate it on the fly.
- Note that both compiled and statically typed languages will have a delay before running for translation and type-checking, respectively.
- Static typing catches errors early, instead of finding them during execution (especially useful for long programs). It’s more “strict” in that it won’t allow for type errors anywhere in your program and often prevents variables from changing types, which further defends against unintended errors.
```cpp
int num = 2
num = '3' // ERROR. Static-type language doesn't allow changing variable type 
```
- Dynamic typing is more flexible (which some appreciate) but allows for variables to change types (sometimes creating unexpected errors).






# terms

- `zero day exploit`: A zero day exploit is a malicious computer attack that takes advantage of a security hole before the vulnerability is known. This means the security issue is made known the same day as the computer attack is released. In other words, the software developer has zero days to prepare for the security breach and must work as quickly as possible to develop a patch or update that fixes the problem.
    - Zero day exploits may involve viruses, trojan horses, worms or other malicious code that can be run within a software program. While most programs do not allow unauthorized code to be executed, hackers can sometimes create files that will cause a program to perform functions unintended by the developer. Programs like Web browsers and media players are often targeted by hackers because they can receive files from the Internet and have access to system functions.
    - While most zero day exploits may not cause serious damage to your system, some may be able to corrupt or delete files. Because the security hole is made known the same day the attack is released, zero day exploits are difficult to prevent, even if you have antivirus software installed on your computer. Therefore, it is always good to keep a backup of your data in a safe place so that no hacker attack can cause you to lose your data.
- `SDK`: Software development kit. SDK stands for software development kit or devkit for short. It’s a set of software tools and programs used by developers to create applications for specific platforms. SDK tools will include a range of things, including libraries, documentation, code samples, processes, and guides that developers can use and integrate into their own apps. SDKs are designed to be used for specific platforms or programming languages. Thus you would need an Android SDK toolkit to build an Android app, an iOS SDK to build an iOS app, a VMware SDK for integrating with the VMware platform, or a Nordic SDK for building Bluetooth or wireless products, and so on.
    - https://www.zhihu.com/question/21691705
- `API`: call a function from a software, without needing to know the code of that function. 
- MS-DOS (disk operating system)
    - A DOS, or disk operating system, is an operating system that runs from a disk drive. The term can also refer to a particular family of disk operating systems, most commonly MS-DOS, an acronym for Microsoft DOS.
    - 
- https://docs.microsoft.com/en-us/cpp/build/reference/dumpbin-reference?redirectedfrom=MSDN&view=msvc-160
- printable strings: 
```python
import string
string.printable # '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
```
- `portability`: whether it can be run in different OS
- `scripting language`: just interpreted language like Bash, Python
- `DUMPBIN.EXE`: The Microsoft COFF Binary File Dumper (DUMPBIN.EXE) displays information about Common Object File Format (COFF) binary files.
    - You can use DUMPBIN to examine COFF object files, standard libraries of COFF objects, executable files, and dynamic-link libraries (DLLs).
- `DLL`: dynamic-link libraries
- `COFF`: Common Object File Format
- `dmg`: DMG files are macOS disk image files
- `regular .exe files`: To understand what makes an app portable, it might be helpful first to take a quick look at how traditional apps get installed in Windows. When you install an app in Windows, the installation files go to several different locations. The bulk of the app’s files are usually copied to a single folder somewhere in the C:\Program Files folder. **Files that contain settings applying to all users of the app may get created in the `ProgramData` folder.**
    - `ProgramData` stores system-wide info, not per-user info!
    - Settings that are particular to **different user accounts** on the PC are stored in files created in the hidden `AppData` folder inside each accounts user folder. **Most apps create entries in the Windows Registry that may also hold various configuration settings**. And many apps take advantage of shared code libraries that get installed with things like the .NET framework and Visual C++ Redistributables.
    - There are distinct advantages to this separation of functions. Multiple apps can share information contained in Registry entries or shared code libraries, preventing unnecessary duplication. Storing user-specific settings in one place and system-wide settings in another means that apps can take better advantage of lots of different Windows features designed for a multi-user system. For starters, each user can rely on their own settings being loaded when they start the app just because they are signed in with their own Windows account. Features like file and share permissions are built on this structure. And, having all program settings saved to designated areas makes backing up your system more reliable.
- `portable application / standalone`:  https://www.howtogeek.com/290358/what-is-a-portable-app-and-why-does-it-matter/
    - A portable app is simply one that doesn’t use an installer. All the files required to run the app reside in a single folder, which you can put anywhere on the system. If you move the folder, the app will still work the same. Instead of installing a portable app, you typically download it as a ZIP file, extract that ZIP to a folder, and run the executable file for the app. **If the app allows you to save settings, those settings are saved in files right inside the same folder**.
    - portable application =/= PE!!!
    - The most significant benefit of using portable apps is self-evident—they’re portable. Stick them on a USB drive, for example, and you can carry them around from computer to computer. They won’t leave any footprint on the PCs you run them on. Everything, including any settings you’ve saved, is saved right in the portable app’s folder on the USB drive. It’s very similar to the way things worked back in the days of MS-DOS and Windows 3.1.
    - Portable apps can be helpful even if you aren’t moving between computers, though. For one thing, they leave a smaller footprint on your PC. They tend to be lighter weight than most installable apps just by virtue of not having to be installed. **You can sync them (along with their settings) to your other PCs using something like Dropbox**. Or, you can just use an app once without having to worry about it leaving cruft on your system.
- `opcode`: machine operation. It is a number interpreted by your machine(virtual or silicon) that represents the operation to perform
    - remember that opcode is a number!!!
    - it is the first portion of the machine code equivalent of a line of assembly code 
- `bytecode`: Same as machine code, except, it's mostly used by a software based interpreter (like Java or CLR)
    - https://opensource.com/article/18/4/introduction-python-bytecode
    - This leaves BYTECODE, which is fundamentally the same as machine code, in that it describes low level operations such as reading and writing memory, and basic calculations. Bytecode is typically conceived to be produced when COMPILING a higher level language, for example PHP or Java, and unlike machine code for many hardware based processors, may have operations to support specific features of the higher level language. A key difference is that the processor of bytecode is usually a program, though processors have been created for interpreting some bytecode specifications, e.g. a processor called SOAR (Smalltalk On A RISC) for Smalltalk bytecode. While you wouldn't typically call native machine code bytecode, for some types of processors such as CISC and EISC (e.g. Linn Rekursiv, from the people who made record players), the processor itself contains a program that is interpreting the machine instructions, so there are parallels.
    - python has `bytecode` function. In fact, we have `.pyc` file 
        - https://opensource.com/article/18/4/introduction-python-bytecode
    - Python source code is compiled into bytecode, the internal representation of a Python program in the CPython interpreter. The bytecode is also cached in .pyc files so that executing the same file is faster the second time (recompilation from source to bytecode can be avoided). This “intermediate language” is said to run on a virtual machine that executes the machine code corresponding to each bytecode. Do note that bytecodes are not expected to work between different Python virtual machines, nor to be stable between Python releases.
    - bytecode is for virtual machine 
    - Bytecode is program code that has been compiled from source code into low-level code designed for a software interpreter. It may be executed by a virtual machine (such as a JVM) or further compiled into machine code, which is recognized by the processor.
    - Different types of bytecode use different syntax, which can be read and executed by the corresponding virtual machine. A popular example is Java bytecode, which is compiled from Java source code and can be run on a Java Virtual Machine (JVM). Below are examples of Java bytecode instructions.
```java
new (create new object)
aload_0 (load reference)
istore (store integer value)
ladd (add long value)
swap (swap two values)
areturn (return value from a function)
```
    - While it is possible to write bytecode directly, it is much more difficult than writing code in a high-level language, like Java. Therefore, bytecode files, such as Java .CLASS files, are most often generated from source code using a compiler, like javac.
    - Bytecode is similar to assembly language in that it is not a high-level language, but it is still somewhat readable, unlike machine language. Both may be considered "intermediate languages" that fall between source code and machine code. The primary difference between the two is that bytecode is generated for a virtual machine (software), while assembly language is created for a CPU (hardware).
- `mnemonic`: https://www.techopedia.com/definition/28287/mnemonic
    - A mnemonic is a term, symbol or name used to define or specify a computing function. Mnemonics are used in computing to provide users with a means to quickly access a function, service or process, bypassing the actual more lengthy method used to perform or achieve it. Assembly language also uses a mnemonic to represent machine operation, or opcode.
    - So, its usually used by assembly language programmers to remember the "OPERATIONS" a machine can do, like "ADD" and "MUL" and "MOV" etc. This is assembler specific.
    - Though better than toggling switches, entering machine code is still slow and error prone. A step up from that is ASSEMBLY CODE, which uses more easily remembered MNEMONICS in place of the actual number that represents an instruction. The job of the ASSEMBLER is primarily to transform the mnemonic form of the program to the corresponding machine code. This makes programming easier, particularly for jump instructions, where part of the instruction is a memory address to jump to or a number of words to skip. Programming in machine code requires painstaking calculations to formulate the correct instruction, and if some code is added or removed, jump instructions may need to be recalculated. The assembler handles this for the programmer.
- `disassembler (反汇编器)`: In programming terminology, to disassemble is to convert a program in its executable (ready-to-run) form (sometimes called object code) into a representation in some form of assembler language so that it is readable by a human.
    - A disassembler is a computer program that translates machine language into assembly language—the inverse operation to that of an assembler. A disassembler differs from a decompiler, which targets a high-level language rather than an assembly language. Disassembly, the output of a disassembler, is often formatted for human-readability rather than suitability for input to an assembler, making it principally a reverse-engineering tool.
    - disassembler: machine code -> assembly code
    - assembler: assembly code -> machine code
    - compiler: source code (e.g., python) -> machine code
    - decompiler: machine code -> source code 
    - interpreters: 
- `assembler`: A program that translates a program written in assembly language into an equivalent program in machine language
- `compiler`: A program called a compiler translates instructions written in high-level languages into machine code.
- `Linker`: A program that combines the object program with other programs in the library and is used in the program to create the executable code.
- `Loader`: A program that loads an executable program into main memory.




# translators (assembler, compiler, interpreter)
- https://www.businessinsider.in/difference-between-compiler-and-interpreter/articleshow/69523408.cms
- https://www.guru99.com/difference-compiler-vs-interpreter.html
- interpreter: https://techterms.com/definition/interpreter
    - The process of compiling a large high-level language program into machine language can take a considerable amount of computer time. Interpreter programs were developed to execute high-level language programs directly (without the need for compilation), although more slowly than compiled programs. Scripting languages such as the popular web languages JavaScript and PHP are processed by interpreters.
    - Interpreters have an advantage over compilers in Internet scripting. An interpreted program can begin executing as soon as it’s downloaded to the client’s machine, without needing to be compiled before it can execute. On the downside, interpreted scripts generally run slower and consume more memory than compiled code.
    - An interpreter is a program that reads and executes code. This includes source code, pre-compiled code, and scripts. Common interpreters include Perl, Python, and Ruby interpreters, which execute Perl, Python, and Ruby code respectively.
    - Interpreters and compilers are similar, since they both recognize and process source code. However, a compiler does not execute the code like an interpreter does. Instead, a compiler simply converts the source code into machine code, which can be run directly by the operating system as an executable program. Interpreters bypass the compilation process and execute the code directly.
    - Since interpreters read and execute code in a single step, they are useful for running scripts and other small programs. Therefore, interpreters are commonly installed on Web servers, which allows developers to run executable scripts within their webpages. These scripts can be easily edited and saved without the need to recompile the code.
    - While interpreters offer several advantages for running small programs, interpreted languages also have some limitations. The most notable is the fact that interpreted code requires an interpreter to run. Therefore, without an interpreter, the source code serves as a plain text file rather than an executable program. Additionally, programs written for an interpreter may not be able to use built-in system functions or access hardware resources like compiled programs can. Therefore, most software applications are compiled rather than interpreted.
- assembler and compiler both outputs machine code, but the input to assembler is assembly language, whereas the input to compiler is high-level language. Both output object file. 
- A computer programmer generates object code with a compiler or assembler. For example, under Linux, the GNU Compiler Collection compiler will generate files with a .o extension which use the ELF format. Compilation on Windows generates files with a .obj extension which use the COFF format. A linker is then used to combine the object code into one executable program or library pulling in precompiled system libraries as needed. JavaScript programs are interpreted while Python and Java programs are compiled into bytecode files.


## object file
- object file is in machine language 
- source code => compiler => object file 
- extension: in Windows => .obj. in Linux => .o 
- CPU can't directly execute object file
- Even though this code is in machine language, it is not in a machine understandable or directly executable form. The object code needs to be linked and then placed in a executable or library file to be able to be executed by the machine.
- On compilation of source code, the machine code generated for different processors like Intel, AMD, an ARM is different. To make code portable, the source code is first converted to Object Code. It is an intermediary code (similar to machine code) that no processor will understand. At run time, the object code is converted to the machine code of the underlying platform.

## executable file and linker
- https://stackoverflow.com/questions/25896207/difference-between-code-object-and-executable-file
- Moreover, the compiler converts the source code to an object file. However, linker links the object files with the system library and combines the object files together to create an executable file.
- After writing the C program, if there are any syntax errors, the programmer should edit them. However, if there are no syntax errors, the compiler converts the source code into an object file. Then the linker performs the linking process. It takes one or more object files generated by the compiler and combines them into a single executable file. Furthermore, it links the other program files and functions the program requires. For example, if the program has the “exp ()” function, the Linker links the program with the math library of the system.
- The programmer does not understand the instructions in the executable file, but the CPU can read and understand those instructions. Therefore, the CPU directly executes the executable file to perform the defined tasks in the program.
- Object files combine together to create an executable file.
- Executable file is the final output, which can be understood by the machine. The object code is linked and placed into a special format which can be understood by the machine (or to be specific, understood by the operating system). The executable file formats are platform dependent.


## object vs executable file
- Object files are source compiled into binary machine language, but they contain unresolved external references (such as printf,for instance). They may need to be linked against other object files, third party libraries and almost always against C/C++ runtime library.
- In Unix, both object and exe files are the same COFF format. The only difference is that object files have unresolved external references, while a.out files don't.

## interpreter vs compiler
- Interpreter translates just one statement of the program at a time into machine code.
    - Compiler scans the entire program and translates the whole of it into machine code at once.
- An interpreter takes very less time to analyze the source code. However, the overall time to execute the process is much slower.
    - A compiler takes a lot of time to analyze the source code. However, the overall time taken to execute the process is much faster.
- An interpreter does not generate an intermediary code. Hence, an interpreter is highly efficient in terms of its memory.
    - A compiler always generates an intermediary object code. It will need further linking. Hence more memory is needed.
- since interpreted languages see code line by line, optimization not as robust as compilers

## programming languages of different levels
- machine language: Any computer can directly understand only its own machine language (also called machine code), defined by its hardware architecture. Machine languages generally consist of numbers (ultimately reduced to 1s and 0s). Such languages are cumbersome for humans.
- assembly language: Programming in machine language was simply too slow and tedious for most programmers. Instead, they began using English-like abbreviations to represent elementary operations. These abbreviations formed the basis of assembly languages. Translator programs called assemblers were developed to convert assembly-language programs to machine language. Although assembly-language code is clearer to humans, it’s incomprehensible to computers until translated to machine language.
- high-level language: To speed up the programming process further, high-level languages were developed in which single statements could be written to accomplish substantial tasks. High-level languages, such as C, C++, Java, C#, Swift and Visual Basic, allow you to write instructions that look more like everyday English and contain commonly used mathematical notations. Translator programs called compilers convert high-level language programs into machine language. The process of compiling a large high-level language program into machine language can take a considerable amount of computer time. Interpreter programs were developed to execute high-level language programs directly (without the need for compilation), although more slowly than compiled programs. Scripting languages such as the popular web languages JavaScript and PHP are processed by interpreters.

## program execution cycle
- source code => compiler => object file => linker => executable 
- `Preprocessor`: Preprocessing occurs before a program is compiled. Some possible actions are inclusion of other files in the file being compiled, definition of symbolic constants and macros, conditional compilation of program code and conditional execution of preprocessing directives. All preprocessing directives begin with #, and only whitespace characters may appear before a preprocessing directive on a line. Preprocessing directives are not C++ statements, so they do not end in a semicolon (;). Preprocessing directives are processed fully before compilation begins.
    - e.g: #include <iostream>
- `compilation`: After processing preprocessor directives, the next step is to verify that the program obeys the rules of the programming language—that is, the program is syntactically correct—and translate the program into the equivalent machine language. The compiler checks the source program for syntax errors and, if no error is found, translates the program into the equivalent machine language. The equivalent machine language program is called an object program.
- `linker`: The programs that you write in a high-level language are developed using an integrated development environment (IDE). The IDE contains many programs that are useful in creating your program. For example, it contains the necessary code (program) to display the results of the program and several mathematical functions to make the programmer’s job somewhat easier. Therefore, if certain code is already available, you can use this code rather than writing your own code. Once the program is developed and successfully compiled, you must still bring the code for the resources used from the IDE into your program to produce a final program that the computer can execute. This prewritten code (program) resides in a place called the library. A program called a linker combines the object program with the programs from libraries.
    - C++ programs typically contain references to functions and data defined elsewhere, such as in the standard libraries or in the private libraries of groups of programmers working on a particular project (Fig. 1.9). The object code produced by the C++ compiler typically contains “holes” due to these missing parts. A linker links the object code with the code for the missing functions to produce an executable program (with no missing pieces). If the program compiles and links correctly, an executable image is produced.

# meaning analysis of program
- syntax (grammar): 
- semantics (meaning): The set of rules that gives meaning to a language is called semantics. For example, the order-of-precedence rules for arithmetic operators are semantic rules. If a program contains syntax errors, the compiler will warn you. What happens when a program contains semantic errors? It is quite possible to eradicate all syntax errors in a program and still not have it run. And if it runs, it may not do what you meant it to do. For example, the following two lines of code are both syntactically correct expressions, but they have different meanings:
```cpp
2+3*5
(2+3)*5
```
- If you substitute one of these lines of code for the other in a program, you will not get the same results—even though the numbers are the same, the semantics are different. You will learn about semantics throughout this book.
- lexical analyzer:
- syntax analyzer:
- semantic analyzer:




## PE files/format
- https://ithelp.ithome.com.tw/articles/10187490
- `PE file` = portable executable 
    - PE file has the following extensions: `.acm, .ax, .cpl, .dll, .drv, .efi, .exe, .mui, .ocx, .scr, .sys, .tsp`
    - PE32 = 32-bit executable file
    - PE32+ = 64-bit executable file
- PE (portable executable) file and file signature: The Windows executable files, also called PE files (such as the files ending with .exe, .dll, .com, .drv, .sys, and so on), have a file signature of `MZ` or hexadecimal characters `4D 5A` in the first two bytes of the file.
- The Portable Executable (PE) file format is used by Windows executables, object code, and DLLs. The PE file format is a data structure that contains the information necessary for the Windows OS loader to manage the wrapped executable code. Nearly every file with executable code that is loaded by Windows is in the PE file format, though some legacy file formats do appear on rare occasion in malware.
- PE files begin with a header that includes information about the code, the type of application, required library functions, and space requirements. The information in the PE header is of great value to the malware analyst.
- refers to malware data science book for details




## executable file types
- binary file
    - already compiled
- executable file: files with extension: `.exe`, `.dll`
- file signature: Most Windows-based malware are executable files ending with extensions such as .exe, .dll, .sys, and so on. But relying on file extensions alone is not recommended. File extension is not the sole indicator of file type. Attackers use different tricks to hide their file by modifying the file extension and changing its appearance to trick users into executing it. Instead of relying on file extension, File signature can be used to determine the file type. 
    - A file signature is a unique sequence of bytes that is written to the file's header. Different files have different signatures, which can be used to identify the type of file. The Windows executable files, also called PE files (such as the files ending with .exe, .dll, .com, .drv, .sys, and so on), have a file signature of MZ or hexadecimal characters 4D 5A in the first two bytes of the file.



## security terms
- https://stackoverflow.com/questions/17638888/difference-between-opcode-byte-code-mnemonics-machine-code-and-assembly
- metadata: last modified, created time, size,...,
- cryptographic hash functions: Most cryptographic hash functions are designed to take a string of any length as input and produce a fixed-length hash value.
    - a mathematical algorithm that maps data of arbitrary size (often called the "message") to a bit array of a fixed size (the "hash value", "hash", or "message digest"). It is a one-way function, that is, a function which is practically infeasible to invert or reverse the computation.[1] Ideally, the only way to find a message that produces a given hash is to attempt a brute-force search of possible inputs to see if they produce a match
- fingerprinting: Fingerprinting involves generating the cryptographic hash values for the suspect binary based on its file content. The cryptographic hashing algorithms such as MD5, SHA1 or SHA256 are considered the de facto standard for generating file hashes for the malware specimens.
    - MD5: The Message-Digest Algorithm 5
    - SHA-1: Secure Hash Algorithm 1
- obfuscation: https://en.wikipedia.org/wiki/Obfuscation_(software)
    - In software development, obfuscation is the deliberate act of creating source or machine code that is difficult for humans to understand.

- File signature: A file signature is a unique sequence of bytes that is written to the file's header. Different files have different signatures, which can be used to identify the type of file. The Windows executable files, also called PE files (such as the files ending with .exe, .dll, .com, .drv, .sys, and so on), have a file signature of MZ or hexadecimal characters 4D 5A in the first two bytes of the file.
- hex editor: The manual method of determining the file type is to look for the file signature by opening it in a hex editor. A hex editor is a tool that allows an examiner to inspect each byte of the file; most hex editors provide many functionalities that help in the analysis of a file.
- file signature: identifiable pieces of known suspicious code (file signatures)
- 
- 

- three feature extraction methods:
    - static
    - dynamic
    - hybrid

- https://pediaa.com/what-is-the-difference-between-object-file-and-executable-file/


## static analysis
- Static analysis is the technique of analyzing the suspect file without executing it. It is an initial analysis method that involves extracting useful information from the suspect binary to make an informed decision on how to classify or analyze it and where to focus your subsequent analysis efforts.
- Static analysis describes the process of analyzing the code or structure of a program to determine its function. The program itself is not run at this time. In contrast, when performing dynamic analysis, the analyst actually runs the program
- Steps to follow:
    - Identifying the malware's target architecture
    - Fingerprinting the malware
    - Scanning the suspect binary with anti-virus engines
    - Extracting strings, functions, and metadata associated with the file
    - Identifying the obfuscation techniques used to thwart analysis
    - Classifying and comparing the malware samples

### determine the malware's target architecture
- During your analysis, determining the file type of a suspect binary will help you identify the malware's target operating system (Windows, Linux, and so on) and architecture (32-bit or 64-bit platforms). For example, if the suspect binary has a file type of Portable Executable (PE), which is the file format for Windows executable files (.exe, .dll, .sys, .drv, .com, .ocx, and so on), then you can deduce that the file is designed to target the Windows operating system.
- Most Windows-based malware are executable files ending with extensions such as .exe, .dll, .sys, and so on. But relying on file extensions alone is not recommended. File extension is not the sole indicator of file type. Attackers use different tricks to hide their file by modifying the file extension and changing its appearance to trick users into executing it. Instead of relying on file extension, `File signature` can be used to determine the file type.

### Fingerprinting the Malware
- Fingerprinting involves generating the cryptographic hash values for the suspect binary
based on its file content. The cryptographic hashing algorithms such as MD5, SHA1 or
SHA256 are considered the de facto standard for generating file hashes for the malware
specimens.

### multiple anti-virus scanning
- Scanning the suspect binary with multiple anti-virus scanners helps in determining whether malicious code signatures exist for the suspect file.
- If a suspect binary does not get detected by the Anti-Virus scanning engines, it does not necessarily mean that the suspect binary is safe. These anti-virus engines rely on signatures and heuristics to detect malicious files. The malware authors can easily modify their code and use obfuscation techniques to bypass these detections, because of which some of the anti-virus engines might fail to detect the binary as malicious.


## dynamic analysis
- This is the process of executing the suspect binary in an isolated environment and monitoring its behavior. This analysis technique is easy to perform and gives valuable insights into the activity of the binary during its execution. This analysis technique is useful but does not reveal all the functionalities of the hostile program

## code analysis
- Code analysis: It is an advanced technique that focuses on analyzing the code to understand the inner workings of the binary. This technique reveals information that is not possible to determine just from static and dynamic analysis. Code analysis is further divided into Static code analysis and Dynamic code analysis. Static code analysis involves disassembling the suspect binary and looking at the code to understand the program's behavior, whereas Dynamic code analysis involves debugging the suspect binary in a controlled manner to understand its functionality. Code analysis requires an understanding of the programming language and operating system concepts. The upcoming chapters (Chapters 4 to 9) will cover the knowledge, tools, and techniques required to perform code analysis.

## Memory analysis (Memory forensics):
- This is the technique of analyzing the computer's RAM for forensic artifacts. It is typically a forensic technique, but integrating it into your malware analysis will assist in gaining an understanding of the malware's behavior after infection. Memory analysis is especially useful to determine the stealth and evasive capabilities of the malware. You will learn how to perform memory analysis in subsequent chapters (Chapters 10 and 11).

## 


- An execution path is a possible flow of control of a program. Each execution path maintains and updates mapping from variables to symbolic expressions during symbolic execution.



## DDOS (Distributed denial-of-service)
- Distributed denial-of-service (DDoS) is a cybersecurity menace which disrupts online services by sending an overwhelming amount of network traffic. These attacks are manually started with botnets that flood the target network. These attacks could have either of the following characteristics:
    - The botnet sends a massive number of requests to the hosting servers.
    - The botnet sends a high volume of random data packets, thus incapacitating the network.
    - 


# paper summary

## Kam1n0:
- assembly code analysis: we have any the binary (executable), and we can use disassembler to transform the binary into its corresponding assembly code. 
    - software plagiarism
    - software patent infringement
- clone search: given a new block of code, we can compare it with the existing code indexed by our engine to determine if it already exists or not. 
- existing work: However, due to the unpredictable effects of different compilers, compiler optimization, and obfuscation techniques, given an unknown function, it is less probable to have a very similar function in the repository.

### questions for paper
- why called Kam1n0?
- why the assembly code is related to graph???
- 


## Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search against Code Obfuscation and Compiler Optimization
- `lexical semantic relationship`: word relationship
- difference between traditional PV-DM and Arm2Vec: 
    - a document is sequentially laid out, which is different than assembly code, because assembly code can be represented as a graph and has a specific syntax.
- Given a binary file, we use the IDA Pro2 disassembler to extract a list of assembly functions, their basic blocks, and control flow graphs.



## static/dynamic approaches
- Dynamic approaches model the semantic similarity by dynamically analyzing the I/O behavior of assembly code 
    - Dynamic approaches are more robust against changes in syntax but less scalable. 
- Static approaches model the similarity between assembly code by looking for their static differences with respect to the syntax or descriptive statistics
    - Static approaches are more scalable and provide better coverage than the dynamic approaches.
- two problems of existing static approach 
    - fail to consider the relationship among features. Assume each the features are independent 
    - The existing static approaches assume that features are equally important or require a mapping of equivalent assembly functions to learn the weights 
        - To solve this problem, we find that it is possible to simulate the way in which an experienced reverse engineer works. Inspired by recent development in representation learning [19], [20], we propose to train a neural network model to read many assembly code data and let the model identify the best representation that distinguishes one function from the rest.


## solution (descriptive)
- Manually specifying all the potential relationships from prior knowledge of assembly language is time consuming and infeasible in practice
- we propose to learn these relationships directly from plain assembly code

## assumptions
- It only needs assembly code as input and does not require any prior knowledge such as the correct mapping between assembly functions.
    - unsupervised learning?
    

### applications:
- help reverse engineers identify assembly clone functions 
- reverse engineers may want to know whether an assembly function already exists in other software. 
    - like google image search, allowing us to identity images similar to our input image. 
- security: locate changed parts, identify known library functions, search for known programming bugs or zero-day vulnerabilities in existing software, detecting software plagiarism, GNU license infringements
- the clone could be due to:
    - complier optimization
    - code obfuscation
- want to embed each assembly function by a vector
- existing method of forming a feature vector:
    - rely on hand-engineered features
    - cons: 
        - fail to consider the relationships between features 
        - fail to identify those unique patterns that can statistically distinguish assembly functions

### problem definition
- four types of clones:
- Type I: literally identical; 
- Type II: syntactically equivalent; 
- Type III: slightly modified;
- Type IV: semantically similar
- focus on type IV
    - e.g.,: the same source code with and without obfuscation, or a patched source code between different releases
- goal: Given an assembly function, our goal is to search for its semantic clones from the repository RP
    - Given a target function ft, the search problem is to retrieve the topk repository functions fs 2 RP, ranked by their semantic similarity, so they can be considered as Type IV clones.

### four-step workflow
1. given a repository of assembly functions, train a neural network to learn their vector representation 
2. produce a vector representation for each function
3. query: given a target function `f_t`, estimate its vector representation
4. find the top-k clones of the vector of `f_t`, using cosine similarity
- The training process is a one-time effort and is efficient
- to learn representation for queries. If a new assembly function is added to the repository, we follow the same procedure in Step 3 to estimate its vector representation. The model can be retrained periodically to guarantee the vectors’ quality.

### difficulty:
- effective search engine is hard, due to compiler optimization and obfuscation. 
- hard to identify semantically similar, but structurally and syntactically different assembly functions as clones 
- assembly code carries richer syntax than plaintext. It contains operations, operands, and control flow that are structurally different than plaintext.
    - sol: an assembly function is represented as a control flow graph. We model the control flow graph as multiple sequences. Each sequence corresponds to a potential execution trace that contains linearly laid-out assembly instructions.
    - for random walk, we start from the root and choose a path from the underlying graph until we reach a leaf
    - PROBLEM: can we result in infinite loop?


### notation
- `d`: dimension of word embedding vector
- `theta_{f_s} (shape = 2xd)`: vector representation of repository function `f_s`.
- `t`: a token (operand / operator)
- `v_t (shape = d)`: vector representation of token `t`. After training, it represents a token’s lexical semantics
- `v'_t (shape = 2xd)`: used for token prediction
- `S(f_s)`: The execution sequences of a repository function `f_s`. `seq[1:i]`, where `seq_i` is one of them. 
- `I(seq_i)`: list of instructions of sequence `i` of a repository function. Use `in_j` to represent one of them
- `A(in_j)`: a list of operands of the instruction `in_j`
    - A => operAnd
- `P(in_j)`: the operation of the instruction `in_j`
    - P => oPeration
- `T(in_j)`: a list of tokens of `in_j`: `T(in_j) = P(in_j) || A(in_j)`
- `CT(in) (shape = 2xd)`: the vector representation of a neighbor instruction in. 
- `f_t`: query assembly function, which is not in RP


### intuition
- The intuition is to use the current function’s vector and the context provided by the neighbor instructions to predict the current instruction
- The vectors provided by neighbor instructions capture the lexical semantic relationship. 
- The function’s vector remembers what cannot be predicted given the context.


### representation of a assembly function
- why need our algo: PV-DM is designed to work with texts that are sequentially laid out. However, for programming languages, we have loops, conditional statements, function calls, recursions, and so an assembly function is generally not sequentially laid-out, and so PV-DM can't be used directly.
- NOTE: in PV-DM, since they use average to combine the vectors, the size of paragraph vector and the word vectors can be the same.
    - in doc2vec: we use concatenation instead of average!
- token: operands and operations in assembly code
- each function `f_s` is mapped to `2xd` vector (`theta_f_s`)
    - it's `2xd` because we need to concatenate the vector for operation and the vector of operands
- each token `t` is mapped to 
    - d-vector (`v_t`): this is the usual embedding for a token `t`
    - 2xd-vector (`v'_t`): for token prediction?
- a function = multiple sequences: `seq_i`
- a sequence = a list of instructions: `in_j`
- an instruction = a list of operands and one operation 
- token = a concatenation of operation and operand (an instruction)
- output: `v'_t`: 2xd. This is for token prediction
- why need the output vector??

### error in paper
- fig4: purple line with `a`

### difference between PV-DM and Asm2Vec
- a document is sequentially laid out
- a function can be represented as a graph and has a specific syntax 
- we want to predict not just a word, but an instruction, which consists of several operands and one operation


### technical
- in PV-DM: assume we have paragraph vector 
- corpus -> paragraphs -> sentences -> words


### aim
- It maximizes the log probability of seeing a token tc at the current instruction, given the current assembly function fs and neighbor instructions.
    - `max log P( t_c | f_s, in_(j-1), in_(j+1) )`
- The intuition is to use the current function’s vector and the context provided by the neighbor instructions to predict the current instruction. 
- The vectors provided by neighbor instructions capture the lexical semantic relationship. 
- The function’s vector (`f_s`) remembers what cannot be predicted given the context. It models the instructions that distinguish the current function from the others.

# points to note
- For an unseen assembly function ft as query ft 2/ RP that does not belong to the set of training assembly functions, we first associate it with a vector ft 2 R2⇥d, which is initialized to a sequence of small values close to zero. Then, we follow the same procedure in the training process, where the neural network goes through each sequence of ft and each instruction of the sequence. In every prediction step, we fix all ~vt and v~0t in the trained model and only propagate errors to ✓~ft. At the end, we have ✓~ft while the vectors for all fs 2 RP and {~vt, v~0t|t 2 D} remain the same. To search for a match, vectors are flattened and compared using cosine similarity.


### questions for paper
- differences between a code function and a sequence of code?
- why use the word repository to mean the set of all assembly functions?
- what is `compiler-output debug symbol`?
- the experiment comparison may not be fair, since competitors are all designed for natural languages, but not assembly language. 
- in experiment, we always use `precision@1`. How about `precision@k`?
- what if we map the all tokens and the function into vectors of the same dimension, then just take the average of the 7 vectors?
- what if we increase the windows size? Now we just look at the previous and next operations
- it seems the vector size `2xd` actually means `2d`..
- what is the meaning of "the training procedure does not require a ground-truth mapping between equivalent assembly functions."??
- for `random edge` operation, we just random one edge and then concatenate the code in the source and target nodes of this edge. 
- can we update our vector representation incrementally?


### finding the top-k results of a query function f_t
- associate `f_t\in\R^{2xd}` with `\theta_f_t`, and initialize to values close to zero
- then just go through the same process of training, except that we only update `theta_f_t`, and keep `v_t` and `v_t'` fixed 
- so, at the end, we have `theta_f_t`, while `f_s`, `v_t`, `v'_t` are all fixed
- finally, find the top-k results using cosine similarity


## suggestion
- in Table, better to have p-values
- 


### experiments
#### setup
- CPU: Intel Xeon 6 core 3.60GHz CPU with 32G memory
- d = 200
- 25 negative samples (k = 25)
- 10 random walks 
- decaying learning rate = 0.025
- metric: precision at position 1 (precision@1)
    - For every query, if a baseline returns no answer, we count the precision as zero. Therefore, Precision@1 captures the ratio of assembly functions that are correctly matched, which is equal to Recall at Position 1.
    - equivalently, equal to 
    - (# functions that are correctly matched with rank 1) / (total number of functions)

#### notes on competitors
- each baseline is configured with the best settings in their paper
- when using PV-DM and PV-DBOW as baselines, each assembly function is treated as a document
    - report the best result among the two 
- state-of-the-art is BinGo and CACompare
- include the Wilcoxon signed-rank test across different binaries to see if the difference in performance is statistically significant.
    - sample size is small.
- 




#### search with different compiler optimization levels
- compiler optimization involves 
    - heavy syntax modifications
    - intensive inlining 
- evaluate based on 10 common libraries.
- compile the libraries using gcc with four different optimization levels (O0,O1,O2,O3)
    - so, we have four different binaries 
- we test each two-combinations of them 
    - for each library, we have 4C2 = 6 combinations
- for each binary from the same library but with different optimization levels, we link their assembly functions using the `compiler-output debug symbols` and generate a clone mapping between functions
- A higher optimization level contains all optimization strategies from the lower level. 
- comparison between O2 and O3 is the easiest one (Figure 6)
    - On average, 26% bytes of a function are modified and none of the functions are identical. 40% of a control flow graph is modified and 65% function pairs share similar graph complexity.
- comparison between O0 and O3 is the most difficult, since the codes are changed substantially
    - On average, 34% bytes of a function are modified and none of the functions are identical. 82% of a control flow graph is modified and 17% function pairs share similar graph complexity.
- in the paper, only comparison of `O2 vs O3` and `O0 vs O3` are shown. This is to illustrate the best and worst situations
- As the difference between two optimization levels increases, the performance of the Asm2Vec decreases.
- The largest binary, OpenSSL, has more than 5,000 functions. Asm2Vec takes on average 153 ms to train an assembly function and 20 ms to process a query. For OpenSSL, CACompare takes on average 12 seconds to fulfill a query.

#### searching with clone obfuscation
- Obfuscator-LLVM (O-LLVM) [24] is built upon the LLVM framework and the CLANG compiler toolchain. It operates at the intermediate language level and modifies a program’s logics before the binary file is generated. It increases the complexity of the binary code. O-LLVM uses three different techniques and their combination: Bogus Control Flow Graph (BCF), Control Flow Flattening (FLA), and Instruction Substitution (SUB).
- we have fewer libraries for comparison because there are compilation errors. 

##### bogus control flow graph (BCF)
- modifies the control flow graph by adding a large number of irrelevant random basic blocks and branches. It will also split, merge, and reorder the original basic blocks. BCF breaks CFG and basic block integrity (on average 149% vertices/edges are added).

##### Control Flow Flattening (FLA)
- reorganizes the original CFG using complex hierarchy of new conditions as switches (see an example in Figure 1). The original instructions are heavily modified to accommodate the new entering conditions and variables. The linear layout has been completely modified (on average 376% vertices and edges are added). Graphbased features are oblivious to this technique. It is also unscalable for a dynamic approach to fully cover the CFG
- https://reverseengineering.stackexchange.com/questions/2221/what-is-a-control-flow-flattening-obfuscation-technique


##### Instruction Substitution (SUB)
- substitutes fragments of assembly code to its equivalent form by going one pass over the function logic using predefined rules. This technique modifies the contents of basic blocks and adds new constants. For example, additions are transformed to a = b $ ($c). Subtractions are transformed to r = rand(); a = b$r; a = a$c; a = a+r. And operations are transformed to a = (b^ v c)&b. SUB does not change much of the graph structure (91% of functions keep the same number of vertex and edge).

#### searching against all binaries
- 

#### Searching Vulnerability Functions







- compile the code using different optimization levels, then see if we can discover that the corresponding functions are the same or not?
- optimization level: O0, O1, O2, O3


### how to deal with non-sequential layout problem
#### selective callee expansion (function inlining)
- selectively inline callee functions in to the caller function. 
- originally used only for dynamic analysis
- now we use it for static analysis 
- BinGo recursively inlines callee, but we only expand the first-order callees in the call graph. Expanding callee functions recursively will include too many callees’ body into the caller, which makes the caller function statically more similar to the callee 
- Let `f_c` be the callee function. We choose to expand a callee function if any of the following conditions are met:
    - `out-deg(f_c)/( out-deg(f_c) + in-deg(f_c) ) >= 0.01` # 
    - `delta(f_s, f_c) = len(f_c)/len(f_s) <=0.6` # if the callee function is longer than or has a comparable length to the caller, the callee will occupy a too large portion of the caller. The expanded function appears similar to the callee. Thus, we add an additional metric to filter out lengthy callees:
    - `len(f_s)<=10` # accommodate wrapper function `f_s`

#### edge coverage
- sample the edges from the callee-expanded control flow graph, until all the edges have been sampled at least once, to ensure that the control flow graph is fully covered 

#### random walk
- another way to generate sequences from a function
- 


# Introduction
- Software reverse engineering is the practice of analyzing a software system to extract design and implementation information. A typical software reverse engineering scenario involves understanding the 
- The purpose of reverse-engineering is to find out how an object or system works.
- Usually the reverse engineers would not have access to the source code of the software under investigation. 
- Therefore, reverse engineers would analyze the binary executable. Or, an alternative would be to use a disassembler to convert binary into assembly code and then analyze the assembly code function by function, and then line by line.
- Although reverse engineering is a time-consuming process, it is nonetheless necessary to understand the inner workings of software that may be suspected to contain malware, vulnerabilities, software infringement or software plagiarism. 



Reverse engineering is a manually intensive but necessary technique for understanding the inner workings of new malware, finding vulnerabilities in existing systems, and detecting patent infringements in released software

# Problem Statement
# Objectives
# Literature Review
# Methodology








