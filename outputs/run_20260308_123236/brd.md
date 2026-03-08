# Business Requirements Document — Car Rental System

## 1. Executive Summary
This document outlines the business requirements for the Car Rental System. The system aims to provide a seamless platform for users to register, log in, browse car tariffs, book vehicles, and manage their rental orders. This system will enhance customer experience and streamline rental operations.

## 2. Business Objectives
1.  To enable new users to easily register and create accounts.
2.  To provide a secure and efficient login process for existing users.
3.  To allow customers to browse available car tariffs and select vehicles.
4.  To facilitate the booking of rental cars for specified periods.
5.  To enable customers to view their historical and current rental orders.
6.  To ensure a user-friendly and intuitive interface for all system interactions.

## 3. Scope

### In Scope
*   User Registration
*   User Login
*   User Logout
*   Car Rental Booking
*   View My Orders
*   View Car Tariffs

### Out of Scope
*   Payment processing
*   Car maintenance tracking
*   Administrative user roles (e.g., for managing car inventory or tariffs)

## 4. Stakeholders

| Role      | Description                                     | Key Interests                                                              |
| :-------- | :---------------------------------------------- | :------------------------------------------------------------------------- |
| Customer  | End-user who rents cars                         | Easy registration, clear tariffs, simple booking, order history            |
| Guest     | Potential customer who can browse and register  | Ability to view tariffs, straightforward registration process              |
| System    | The Car Rental System application               | Data integrity, secure authentication, functional workflows                |
| Developer | Responsible for building and maintaining system | Clear requirements, technical feasibility, maintainable code               |

## 5. Business Requirements

**BR-001: User Registration**
*   Description: The system shall allow new users to register an account by providing necessary details.
*   Priority: High
*   Acceptance: A new user can successfully create an account via `registration.php` and have their details stored in the `registerhere` table.

**BR-002: User Login**
*   Description: The system shall allow registered users to log in securely using their credentials.
*   Priority: High
*   Acceptance: A registered user can successfully log in via `login.php` and `session.php`, establishing an authenticated session using `$_SESSION`.

**BR-003: User Logout**
*   Description: The system shall allow logged-in users to securely log out of their session.
*   Priority: Low
*   Acceptance: A logged-in user can initiate a logout process via `logout.php`, terminating their active `$_SESSION`.

**BR-004: Car Rental Booking**
*   Description: The system shall enable logged-in users to book a car for a specified rental period.
*   Priority: High
*   Acceptance: A logged-in user can select a car, specify rental dates, and submit a booking request via `rent.php` and `request.php`, which is then recorded in the `request` table.

**BR-005: View My Orders**
*   Description: The system shall allow logged-in users to view a list of their past and current car rental orders.
*   Priority: High
*   Acceptance: A logged-in user can access their rental order history via `myorder.php`, displaying data retrieved from the `request` and `registerhere` tables.

**BR-006: View Car Tariffs**
*   Description: The system shall display available car information and their associated rental rates to all users.
*   Priority: Medium
*   Acceptance: All users, logged in or as guests, can view car tariff information by accessing `tarrif.php`.

## 6. Data Requirements
The `registerhere` table stores user account information, including unique usernames, passwords, email addresses, and phone numbers. The `request` table stores details of car rental bookings, linking users to specific cars, rental dates, and other relevant order information such as distance, driver's license number, credit card details, and mobile number.

## 7. Business Rules
1.  Usernames must be unique across all registered users.
2.  Phone numbers must be validated using the `validate_phone_number` function during registration.
3.  User login credentials must match an existing registered user in the `registerhere` table.
4.  Car IDs used in rental requests must correspond to valid cars available in the system (implied, not explicitly defined in provided tables).
5.  Dates of issue and return for car rentals must be valid and logically sequenced.
6.  Rental requests must capture distance, driver's license number, credit card details, and mobile number.
7.  Only users who are logged in (indicated by a valid `$_SESSION`) can view their rental orders via `myorder.php`.

## 8. Assumptions and Constraints
*   **Assumption:** The system assumes the existence of a `validate_phone_number` function.
*   **Assumption:** The system assumes a mechanism for managing car inventory and their associated tariffs, although not detailed in the provided scope.
*   **Constraint:** The system is built on a raw PHP application without a framework.
*   **Constraint:** Authentication relies solely on `$_SESSION` variables.
*   **Constraint:** Data persistence is managed through direct SQL queries to defined tables.

## 9. Glossary

| Term           | Definition                                                              |
| :------------- | :---------------------------------------------------------------------- |
| User           | An individual who interacts with the Car Rental System.                 |
| Customer       | A registered User who can book cars and view orders.                    |
| Guest          | A non-registered User who can view tariffs and register.                |
| RentalRequest  | A record of a car booking made by a User.                               |
| Car            | A vehicle available for rent within the system.                         |
| User Management| The bounded context responsible for user registration, login, and logout. |
| Booking        | The bounded context responsible for car rental requests and management. |
| Catalog        | The bounded context responsible for displaying car tariffs.             |
| `$_SESSION`    | A PHP mechanism for storing user session data, used for authentication. |