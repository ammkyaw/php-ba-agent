# Software Requirements Specification — Car Rental System

## 1. Introduction
### 1.1 Purpose
This document specifies the software requirements for the Car Rental System. It details the functional and non-functional requirements, user characteristics, and system constraints. This SRS is intended for use by developers, testers, and project managers involved in the development of the Car Rental System.

### 1.2 System Overview
The Car Rental System is a web-based application designed to allow users to rent cars. It facilitates user registration, car browsing, booking, and management of rental orders. The system supports two primary user roles: Customer and Guest.

### 1.3 Definitions and Abbreviations
*   **SRS**: Software Requirements Specification
*   **UI**: User Interface
*   **DB**: Database
*   **PDO**: PHP Data Objects
*   **mysqli**: MySQL Improved Extension
*   **$_SESSION**: PHP superglobal array for session variables
*   **$_POST**: PHP superglobal array for HTTP POST variables
*   **$_GET**: PHP superglobal array for HTTP GET variables

## 2. Overall Description
### 2.1 System Context
The Car Rental System is a standalone web application. It interacts with a relational database for data persistence. Users access the system via a web browser.

### 2.2 User Classes and Characteristics

*   **Role**: Customer
    *   **Technical Level**: Intermediate. Familiar with web applications and online forms.
    *   **Frequency of Use**: Moderate to High, depending on rental needs.
    *   **Key Tasks**: Register, log in, view car tariffs, book cars, view rental orders, log out.

*   **Role**: Guest
    *   **Technical Level**: Basic. Familiar with web browsing.
    *   **Frequency of Use**: Low, primarily for initial exploration or registration.
    *   **Key Tasks**: View car tariffs, register.

### 2.3 Operating Environment
The system shall operate on a web server environment supporting PHP and a MySQL database. Client access will be via standard web browsers (e.g., Chrome, Firefox, Safari, Edge).

### 2.4 Assumptions and Dependencies
*   Users have access to a stable internet connection.
*   The underlying web server and database server are operational.
*   The `validate_phone_number` function is available and correctly implemented.

## 3. Functional Requirements

### 3.1 User Registration
**FR-001**: The system shall allow guests to register for a new customer account.
    *   **Input**:
        *   `fname` (string, required)
        *   `lname` (string, required)
        *   `phone` (string, required, validated by `validate_phone_number` function)
        *   `M` (string, gender selection, e.g., 'M', 'F', 'O')
        *   `dob` (date, required)
        *   `email` (string, required, email format)
        *   `uname` (string, required, unique)
        *   `pwd` (string, required, hashed before storage)
        *   `city` (string, required)
        *   `state` (string, required)
        *   `country` (string, required)
    *   **Processing**:
        *   Validate all required fields.
        *   Validate phone number format using `validate_phone_number`.
        *   Check for username uniqueness in the `registerhere` table.
        *   Hash the password before insertion.
        *   Insert new user record into the `registerhere` table.
    *   **Output**: A new user account is created. The user is redirected to `login.php`.
    *   **Pages**: `registration.php`
    *   **Tables**: `registerhere`

### 3.2 User Login
**FR-002**: The system shall allow registered customers to log in.
    *   **Input**:
        *   `uname` (string, required)
        *   `pwd` (string, required)
    *   **Processing**:
        *   Retrieve user record from `registerhere` table based on `uname`.
        *   Verify the provided `pwd` against the stored hashed password.
        *   If credentials match, set `$_SESSION['uname']` to the username.
    *   **Output**: User is authenticated. If successful, the user is redirected to `rent.php`. If unsuccessful, an error message is displayed, and the user remains on `login.php`.
    *   **Pages**: `login.php`, `session.php`
    *   **Tables**: `registerhere`

### 3.3 User Logout
**FR-003**: The system shall allow logged-in customers to log out.
    *   **Input**: None.
    *   **Processing**:
        *   Destroy the user's session using `session_destroy()`.
        *   Unset relevant session variables (e.g., `$_SESSION['uname']`).
    *   **Output**: User session is terminated. The user is redirected to `login.php`.
    *   **Pages**: `logout.php`
    *   **Tables**: None

### 3.4 Car Rental Booking
**FR-004**: The system shall allow logged-in customers to book a car.
    *   **Input**:
        *   `carid` (integer, required, from `rent.php` selection)
        *   `doi` (date, required, Date of Issue)
        *   `dor` (date, required, Date of Return, must be after `doi`)
        *   `distance` (float, required)
        *   `dln` (string, required, Driver's License Number)
        *   `cc` (string, required, Credit Card Number)
        *   `dc` (string, required, Credit Card Expiry Date)
        *   `no` (string, required, Number of passengers)
        *   `mob` (string, required, Mobile Number)
        *   `sub` (string, submit button value, from `request.php`)
    *   **Processing**:
        *   Validate all required input fields.
        *   Validate `doi` and `dor` dates, ensuring `dor` is after `doi`.
        *   Retrieve customer's username from `$_SESSION['uname']`.
        *   Insert a new rental request record into the `request` table, including `carid`, `doi`, `dor`, `distance`, `dln`, `cc`, `dc`, `no`, `mob`, and the customer's username.
    *   **Output**: A car rental request is submitted and stored in the `request` table. The user is redirected to `rent.php`.
    *   **Pages**: `rent.php`, `request.php`
    *   **Tables**: `request`

### 3.5 View My Orders
**FR-005**: The system shall allow logged-in customers to view their rental orders.
    *   **Input**: None.
    *   **Processing**:
        *   Retrieve the logged-in customer's username from `$_SESSION['uname']`.
        *   Query the `request` table for all records associated with the customer's username.
        *   Join with the `registerhere` table to potentially display customer details alongside order details (though not explicitly stated in the domain model, this is a common pattern).
    *   **Output**: A list of the user's rental orders is displayed on the `myorder.php` page.
    *   **Pages**: `myorder.php`
    *   **Tables**: `request`, `registerhere`

### 3.6 View Car Tariffs
**FR-006**: The system shall display available car information and rental rates to all users (Guests and Customers).
    *   **Input**: None.
    *   **Processing**:
        *   Retrieve car information and associated tariff details. (Database table for cars/tariffs is not explicitly defined in the provided model, but implied by the feature).
    *   **Output**: A list of cars with their details and rental information is displayed on the `tarrif.php` page.
    *   **Pages**: `tarrif.php`
    *   **Tables**: (Implied, e.g., `cars`, `tariffs`)

## 4. Non-Functional Requirements
### 4.1 Security Requirements
*   **SR-001**: User passwords shall be stored securely using a one-way hashing algorithm.
*   **SR-002**: Session management shall be implemented to protect against session hijacking.
*   **SR-003**: Input validation shall be performed on all user-submitted data to prevent injection attacks.

### 4.2 Performance Requirements
*   **PR-001**: Page load times for all pages shall not exceed 3 seconds under normal network conditions.
*   **PR-002**: Database queries for displaying lists (e.g., tariffs, orders) shall execute within 1 second.

### 4.3 Usability Requirements
*   **UR-001**: The user interface shall be intuitive and easy to navigate for users with basic web experience.
*   **UR-002**: Error messages shall be clear and provide guidance for resolution.

### 4.4 Reliability Requirements
*   **RR-001**: The system shall be available 99.5% of the time, excluding scheduled maintenance.
*   **RR-002**: Data integrity shall be maintained; rental bookings and user registrations shall be reliably stored.

## 5. External Interface Requirements
### 5.1 User Interface
The system will be accessed via a web browser. Key page flows include:
*   **Guest**: `tarrif.php` -> `registration.php` -> `login.php`
*   **Customer**: `login.php` -> `session.php` (authentication) -> `rent.php` -> `request.php` (booking) -> `myorder.php` (view orders) -> `logout.php` (logout)
The UI will consist of forms for data entry (registration, booking) and display pages for information (tariffs, orders).

### 5.2 Database Interface
The system will interact with a relational database using PDO or mysqli. The following tables are inferred:

*   **`registerhere`**: Stores user registration information.
    *   `id` (INT, Primary Key)
    *   `fname` (VARCHAR)
    *   `lname` (VARCHAR)
    *   `phone` (VARCHAR)
    *   `gender` (VARCHAR, inferred from 'M' input)
    *   `dob` (DATE)
    *   `email` (VARCHAR)
    *   `uname` (VARCHAR, Unique)
    *   `pwd` (VARCHAR, Hashed Password)
    *   `city` (VARCHAR)
    *   `state` (VARCHAR)
    *   `country` (VARCHAR)

*   **`request`**: Stores car rental booking requests.
    *   `id` (INT, Primary Key)
    *   `carid` (INT, Foreign Key to a car table)
    *   `doi` (DATE)
    *   `dor` (DATE)
    *   `distance` (DECIMAL or FLOAT)
    *   `dln` (VARCHAR, Driver's License Number)
    *   `cc` (VARCHAR, Credit Card Number)
    *   `dc` (VARCHAR, Credit Card Expiry Date)
    *   `no` (INT, Number of passengers)
    *   `mob` (VARCHAR, Mobile Number)
    *   `customer_uname` (VARCHAR, Foreign Key to `registerhere.uname`)

*   **(Implied) `cars`**: Stores car details.
    *   `id` (INT, Primary Key)
    *   `model` (VARCHAR)
    *   `make` (VARCHAR)
    *   `year` (INT)
    *   `rental_rate` (DECIMAL)

*   **(Implied) `tariffs`**: Stores rental rates, potentially linked to cars.
    *   `id` (INT, Primary Key)
    *   `car_id` (INT, Foreign Key to `cars.id`)
    *   `rate_per_day` (DECIMAL)

## 6. System Constraints
*   **SC-001**: The system must be implemented using PHP.
*   **SC-002**: Database interactions must use PDO or mysqli.
*   **SC-003**: User authentication must rely on `$_SESSION` variables.
*   **SC-004**: Form data retrieval must use `$_POST` or `$_GET`.
*   **SC-005**: Navigation between pages must be handled via `header('Location: ...')` redirects.