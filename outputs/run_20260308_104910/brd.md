# Business Requirements Document — Car Rental System

## 1. Executive Summary
This document outlines the business requirements for the Car Rental System, a web-based application designed to facilitate car rentals for customers and manage rental processes. The system will enable user registration, car booking, and order management, providing a streamlined experience for both users and administrators.

## 2. Business Objectives
1. To provide a user-friendly platform for customers to rent vehicles.
2. To streamline the car booking and rental request process.
3. To enable customers to view their rental history and current bookings.
4. To ensure secure user authentication and data management.
5. To offer a clear display of available car tariffs and rental rates.
6. To support administrative oversight of car inventory and user accounts.

## 3. Scope
### In Scope
*   User Registration
*   User Login
*   Car Rental Booking
*   View My Orders
*   User Logout
*   View Car Tariffs

### Out of Scope
*   System Administrator specific interfaces for managing car data and user accounts.
*   Payment processing integration beyond data collection.
*   Automated email notifications.

## 4. Stakeholders

| Role              | Description                                                                 | Key Interests                                                              |
| :---------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------- |
| Customer          | End-user of the system who rents cars.                                      | Easy booking, clear pricing, access to rental history.                     |
| System Administrator | Manages car inventory, user accounts, and system configuration (implicit). | System integrity, accurate data, efficient user management.                |
| Business Owner    | Oversees the car rental operations and system profitability.                | Customer satisfaction, revenue generation, operational efficiency.         |

## 5. Business Requirements

**BR-001: User Registration**
*   Description: The system shall allow new users to create an account by providing necessary details to register.
*   Priority: High
*   Acceptance: New users can successfully create an account via `registration.php` and their details are stored in the `registerhere` table.

**BR-002: User Login**
*   Description: The system shall allow registered users to securely log in to access their accounts and system features.
*   Priority: High
*   Acceptance: Authenticated users can access restricted features after logging in via `login.php` and `session.php`, with their session maintained via `$_SESSION`.

**BR-003: Car Rental Booking**
*   Description: The system shall enable authenticated users to select an available car and book it for a specified rental period.
*   Priority: High
*   Acceptance: Authenticated users can submit a rental request via `rent.php` and `request.php`, with booking details including `carid`, `doi`, and `dor` stored in the `request` table.

**BR-004: View My Orders**
*   Description: The system shall provide authenticated users with the ability to view a list of their past and current rental orders.
*   Priority: High
*   Acceptance: Authenticated users can view their rental history by accessing `myorder.php`, which retrieves data from the `request` and `registerhere` tables.

**BR-005: User Logout**
*   Description: The system shall allow authenticated users to securely log out of their session.
*   Priority: Low
*   Acceptance: Users can terminate their session by navigating to `logout.php`.

**BR-006: View Car Tariffs**
*   Description: The system shall display available car information and their corresponding rental rates to all users.
*   Priority: Medium
*   Acceptance: Users can view car tariffs and rates by accessing `tarrif.php`.

## 6. Data Requirements
The `registerhere` table stores user account information, including unique usernames (acting as primary keys), passwords, and contact details such as phone numbers. The `request` table is used to record rental bookings, linking users to specific cars and storing rental period details (dates of issue and return), along with associated charges and payment information.

## 7. Business Rules
1.  Username must be unique (primary key in `registerhere`).
2.  Phone number format must be validated using the `validate_phone_number` function.
3.  User login credentials must match a registered user in the `registerhere` table.
4.  Car rental bookings require a selected `carid`.
5.  Car rental bookings require specified start and end dates (`doi`, `dor`).
6.  Payment details (`cc`, `dc`) and other charges (`distance`, `no`, `mob`, `sub`) are collected during the booking process.
7.  Only authenticated users (indicated by `$_SESSION` variables) can view their rental orders.

## 8. Assumptions and Constraints
*   **Assumption:** The system assumes the existence of a `validate_phone_number` function for phone number validation.
*   **Assumption:** Implicit administrative functions for managing car data and user accounts are handled outside the scope of the provided evidence.
*   **Constraint:** The application is a raw PHP application and does not utilize a framework.
*   **Constraint:** Authentication is managed using `$_SESSION` variables.

## 9. Glossary

| Term           | Definition                                                              |
| :------------- | :---------------------------------------------------------------------- |
| BRD            | Business Requirements Document                                          |
| User           | An individual interacting with the Car Rental System.                   |
| Customer       | A user role with the ability to register, book cars, and view orders.   |
| System Admin   | An implicit role responsible for managing system data and users.        |
| Registration   | The process of creating a new user account in the system.               |
| Login          | The process of authenticating an existing user to access the system.    |
| Booking        | The act of reserving a car for a specific period.                       |
| Tariff         | Information detailing car availability and rental pricing.              |
| `registerhere` | Database table storing user registration information.                   |
| `request`      | Database table storing car rental booking details.                      |
| `$_SESSION`    | PHP mechanism for storing user session data, used for authentication. |
| `doi`          | Date of Issue (rental start date).                                      |
| `dor`          | Date of Return (rental end date).                                       |