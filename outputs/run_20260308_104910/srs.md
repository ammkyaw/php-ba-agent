# Software Requirements Specification — Car Rental System

## 1. Introduction
### 1.1 Purpose
This document specifies the software requirements for the Car Rental System. It is intended for use by developers, testers, and project managers involved in the development of the system.

### 1.2 System Overview
The Car Rental System is a web-based application designed to facilitate car rentals for users. It provides functionalities for user registration, login, car booking, viewing rental orders, and viewing car tariffs. The system aims to streamline the car rental process for both customers and administrators.

### 1.3 Definitions and Abbreviations
*   **SRS**: Software Requirements Specification
*   **UI**: User Interface
*   **DB**: Database
*   **PDO**: PHP Data Objects
*   **mysqli**: MySQL Improved Extension

## 2. Overall Description
### 2.1 System Context
The Car Rental System is a standalone web application. It interacts with a relational database for data persistence. Users access the system via a web browser.

### 2.2 User Classes and Characteristics

*   **Role**: Customer
    *   **Technical Level**: Basic to Intermediate understanding of web applications.
    *   **Frequency of Use**: Varies based on rental needs.
    *   **Key Tasks**: Register, log in, view car tariffs, book cars, view rental orders, log out.

*   **Role**: System Administrator
    *   **Technical Level**: Intermediate to Advanced understanding of web applications and database management.
    *   **Frequency of Use**: As needed for system maintenance.
    *   **Key Tasks**: Implicitly manages car data and user accounts (no explicit admin pages provided in the evidence).

### 2.3 Operating Environment
The system shall operate on a web server with PHP and a MySQL database. Client access will be through standard web browsers.

### 2.4 Assumptions and Dependencies
*   Users have a stable internet connection to access the system.
*   The database server is available and operational.
*   The system relies on PHP sessions for maintaining user authentication state.

## 3. Functional Requirements

### 3.1 User Registration
**FR-REG-001**: The system shall allow new users to register for an account.
*   **Input**:
    *   `fname` (First Name): String, required.
    *   `lname` (Last Name): String, required.
    *   `phone` (Phone Number): String, required. Must pass `validate_phone_number` function.
    *   `M` (Gender): String, required (assumed to be 'M' or 'F' or similar).
    *   `dob` (Date of Birth): String, required (date format validation implied).
    *   `email` (Email Address): String, required (email format validation implied).
    *   `uname` (Username): String, required. Must be unique.
    *   `pwd` (Password): String, required.
    *   `city` (City): String, required.
    *   `state` (State): String, required.
    *   `country` (Country): String, required.
*   **Processing**:
    *   Validate input fields for completeness and format.
    *   Check if the provided `uname` already exists in the `registerhere` table.
    *   If `uname` is unique, insert the new user's details into the `registerhere` table.
*   **Output**: A new user account is created in the `registerhere` table. The user is likely redirected to a confirmation or login page (specific redirect not detailed).
*   **Pages**: `registration.php`
*   **Tables**: `registerhere`

### 3.2 User Login
**FR-LOGIN-001**: The system shall allow registered users to log in.
*   **Input**:
    *   `uname` (Username): String, required.
    *   `pwd` (Password): String, required.
*   **Processing**:
    *   Retrieve user record from the `registerhere` table where `uname` matches the input.
    *   Verify if the provided `pwd` matches the stored password for the user.
    *   If credentials match, establish a user session by setting `$_SESSION['uname']` to the logged-in username.
*   **Output**: User is authenticated and redirected to `rent.php`.
*   **Pages**: `login.php`, `session.php`
*   **Tables**: `registerhere`

### 3.3 Car Rental Booking
**FR-BOOK-001**: The system shall allow authenticated users to book a car.
*   **Input**:
    *   `carid` (Car ID): Integer, required.
    *   `doi` (Date of Issue): String, required (date format validation implied).
    *   `dor` (Date of Return): String, required (date format validation implied).
    *   `distance` (Distance): Numeric, required.
    *   `dln` (Driver's License Number): String, required.
    *   `cc` (Credit Card Number): String, required (format validation implied).
    *   `dc` (Credit Card Expiry Date): String, required (date format validation implied).
    *   `no` (Number of Passengers): Integer, required.
    *   `mob` (Mobile Number): String, required.
    *   `sub` (Subscription/Discount Code): String, optional.
*   **Processing**:
    *   Verify that the user is authenticated (check for `$_SESSION['uname']`).
    *   Validate all input fields for completeness and correct format.
    *   Insert a new rental request record into the `request` table with the provided details and the authenticated user's username.
*   **Output**: A rental request is recorded in the `request` table. The user is redirected to `rent.php`.
*   **Pages**: `rent.php`, `request.php`
*   **Tables**: `request`

### 3.4 View My Orders
**FR-ORDER-001**: The system shall allow authenticated users to view their rental orders.
*   **Input**: None.
*   **Processing**:
    *   Verify that the user is authenticated (check for `$_SESSION['uname']`).
    *   Query the `request` table to retrieve all rental requests associated with the authenticated user's username.
    *   Join with the `registerhere` table if necessary to display user-specific information alongside order details.
*   **Output**: A list of the user's rental requests is displayed on the `myorder.php` page.
*   **Pages**: `myorder.php`
*   **Tables**: `request`, `registerhere`

### 3.5 User Logout
**FR-LOGOUT-001**: The system shall allow authenticated users to log out.
*   **Input**: None.
*   **Processing**:
    *   Destroy the user's session.
    *   Unset all session variables, including `$_SESSION['uname']`.
*   **Output**: User session is terminated, and the user is redirected to `login.php`.
*   **Pages**: `logout.php`
*   **Tables**: None

### 3.6 View Car Tariffs
**FR-TARIFF-001**: The system shall display available car information and rental rates.
*   **Input**: None.
*   **Processing**:
    *   Retrieve car data and associated tariff information (specific table not provided in evidence, assumed to exist).
*   **Output**: A list of cars with their details and pricing is displayed on the `tarrif.php` page.
*   **Pages**: `tarrif.php`
*   **Tables**: None (assumed, as no specific car tariff table was provided in the evidence).

## 4. Non-Functional Requirements
### 4.1 Security Requirements
*   Passwords shall be stored securely (e.g., using hashing).
*   User sessions shall be managed securely to prevent unauthorized access.
*   Input validation shall be performed to prevent common web vulnerabilities (e.g., SQL injection, XSS).

### 4.2 Performance Requirements
*   Page load times shall be within acceptable limits (e.g., under 3 seconds for typical operations).
*   Database queries shall be optimized for efficient retrieval of data.

### 4.3 Usability Requirements
*   The user interface shall be intuitive and easy to navigate.
*   Error messages shall be clear and informative.
*   The system shall provide consistent feedback to users on their actions.

### 4.4 Reliability Requirements
*   The system shall be available during business hours with minimal downtime.
*   Data integrity shall be maintained through proper database transactions.

## 5. External Interface Requirements
### 5.1 User Interface
The system will be accessed via a web browser. Key page flows include:
*   **Registration**: `registration.php` -> (Success/Error Message) -> `login.php`
*   **Login**: `login.php` -> `session.php` -> (Success) `rent.php` or (Failure) `login.php`
*   **Booking**: `rent.php` -> `request.php` -> (Success) `rent.php`
*   **View Orders**: `myorder.php` (accessible after login)
*   **Logout**: `logout.php` -> `login.php`
*   **View Tariffs**: `tarrif.php` (accessible without login)

### 5.2 Database Interface
The system interacts with a MySQL database. Key tables inferred from the evidence include:

*   **`registerhere`**: Stores user registration information.
    *   `uname` (VARCHAR, PRIMARY KEY): Unique username.
    *   `pwd` (VARCHAR): Hashed password.
    *   `fname` (VARCHAR): First name.
    *   `lname` (VARCHAR): Last name.
    *   `phone` (VARCHAR): Phone number.
    *   `M` (CHAR): Gender indicator.
    *   `dob` (DATE): Date of birth.
    *   `email` (VARCHAR): Email address.
    *   `city` (VARCHAR): City.
    *   `state` (VARCHAR): State.
    *   `country` (VARCHAR): Country.

*   **`request`**: Stores car rental booking requests.
    *   `carid` (INT): Identifier for the rented car.
    *   `doi` (DATE): Date of issue (rental start date).
    *   `dor` (DATE): Date of return (rental end date).
    *   `distance` (DECIMAL): Estimated distance for the rental.
    *   `dln` (VARCHAR): Driver's license number.
    *   `cc` (VARCHAR): Credit card number.
    *   `dc` (DATE): Credit card expiry date.
    *   `no` (INT): Number of passengers.
    *   `mob` (VARCHAR): Mobile number.
    *   `sub` (VARCHAR): Subscription or discount code.
    *   `uname` (VARCHAR): Username of the customer who made the request (foreign key to `registerhere.uname`).

## 6. System Constraints
*   The system must be implemented using PHP and a MySQL database.
*   Session management must rely on PHP's built-in session handling.
*   Direct database access via PDO or mysqli is required.
*   The system must adhere to the specified page names and input field names.