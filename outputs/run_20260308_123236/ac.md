# Acceptance Criteria — Car Rental System

## Overview
Acceptance testing for the Car Rental System will focus on verifying that each feature functions according to its defined requirements and business rules. Testing will be performed by simulating user interactions through the specified pages and validating the resulting system state, including database entries and user redirections. Both positive and negative scenarios will be covered to ensure robustness.

---

## AC-01: User Registration
**Feature:** Allows new users to create an account in the system.
**Pages:** `registration.php`
**Tables:** `registerhere`

### Acceptance Criteria:

**AC-01-01: Successful User Registration**
- Given a guest user is on the `registration.php` page
- When the user fills in all required fields with valid data (fname, lname, phone, M, dob, email, uname, pwd, city, state, country) and submits the form
- Then a new row is inserted into the `registerhere` table with the provided user details
- And the user is redirected to `login.php`

**AC-01-02: Registration with Duplicate Username**
- Given a guest user is on the `registration.php` page
- And a user with username "existinguser" already exists in the `registerhere` table
- When the user attempts to register with "existinguser" as the username and other valid details
- Then an error message indicating "Username already exists" is displayed
- And no new row is inserted into the `registerhere` table

**AC-01-03: Registration with Invalid Phone Number**
- Given a guest user is on the `registration.php` page
- When the user fills in all required fields with valid data except for an invalid phone number (e.g., "123") and submits the form
- Then an error message indicating "Invalid phone number" is displayed (assuming `validate_phone_number` function returns false)
- And no new row is inserted into the `registerhere` table

**AC-01-04: Registration with Missing Required Field**
- Given a guest user is on the `registration.php` page
- When the user fills in all fields except for the `email` field and submits the form
- Then an error message indicating "Email is required" is displayed
- And no new row is inserted into the `registerhere` table

---

## AC-02: User Login
**Feature:** Allows existing users to log in to the system.
**Pages:** `login.php`, `session.php`
**Tables:**

### Acceptance Criteria:

**AC-02-01: Successful User Login**
- Given a registered user with username "testuser" and password "password123" exists
- And the user is on the `login.php` page
- When the user enters "testuser" in the `uname` field and "password123" in the `pwd` field and submits the form
- Then the `$_SESSION["user_id"]` is set with the user's ID
- And the user is redirected to `rent.php`

**AC-02-02: Login with Invalid Credentials**
- Given a user is on the `login.php` page
- When the user enters an incorrect username "wronguser" or an incorrect password "wrongpass" and submits the form
- Then an error message "Invalid username or password" is displayed
- And the `$_SESSION["user_id"]` is not set

**AC-02-03: Login Attempt by Unregistered User**
- Given a user is on the `login.php` page
- When the user enters a username "nonexistentuser" that does not exist in the system and a password "anypassword" and submits the form
- Then an error message "Invalid username or password" is displayed
- And the `$_SESSION["user_id"]` is not set

---

## AC-03: User Logout
**Feature:** Allows logged-in users to log out of the system.
**Pages:** `logout.php`
**Tables:**

### Acceptance Criteria:

**AC-03-01: Successful User Logout**
- Given a logged-in user (i.e., `$_SESSION["user_id"]` is set)
- When the user navigates to `logout.php`
- Then the user's session is destroyed
- And the user is redirected to `login.php`

**Complexity: Low — single operation**

---

## AC-04: Car Rental Booking
**Feature:** Enables logged-in users to book a car for a specified period.
**Pages:** `rent.php`, `request.php`
**Tables:** `request`

### Acceptance Criteria:

**AC-04-01: Successful Car Rental Booking**
- Given a logged-in customer is on the `rent.php` page
- When the user selects a `carid`, enters valid `doi` (Date of Issue), `dor` (Date of Return), `distance`, `dln` (Driver's License Number), `cc` (Credit Card details), `dc` (Discount Code - optional), `no` (Number of days), `mob` (Mobile number) and submits the booking request via `request.php`
- Then a new row is inserted into the `request` table with the booking details
- And the user is redirected back to `rent.php`

**AC-04-02: Booking with Invalid Dates**
- Given a logged-in customer is on the `rent.php` page
- When the user selects a `carid`, enters an invalid `doi` (e.g., past date) or `dor` (e.g., before `doi`) and submits the booking request via `request.php`
- Then an error message indicating "Invalid dates" is displayed
- And no new row is inserted into the `request` table

**AC-04-03: Booking with Missing Required Fields**
- Given a logged-in customer is on the `rent.php` page
- When the user selects a `carid`, enters valid `doi` and `dor`, but omits the `dln` (Driver's License Number) and submits the booking request via `request.php`
- Then an error message indicating "Driver's License Number is required" is displayed
- And no new row is inserted into the `request` table

**AC-04-04: Booking with Invalid Car ID**
- Given a logged-in customer is on the `rent.php` page
- When the user attempts to book a car with an invalid `carid` (e.g., non-existent ID) and submits the booking request via `request.php`
- Then an error message indicating "Invalid Car ID" is displayed
- And no new row is inserted into the `request` table

---

## AC-05: View My Orders
**Feature:** Allows logged-in users to view their past and current car rental orders.
**Pages:** `myorder.php`
**Tables:** `request`, `registerhere`

### Acceptance Criteria:

**AC-05-01: View Orders for Logged-in User**
- Given a logged-in customer with existing rental orders in the `request` table
- When the user navigates to `myorder.php`
- Then a list of the customer's rental orders is displayed, showing details from both `request` and `registerhere` tables (e.g., car details, dates, user information)

**AC-05-02: View Orders for User with No Orders**
- Given a logged-in customer who has not made any rental bookings
- When the user navigates to `myorder.php`
- Then a message indicating "You have no rental orders." is displayed

**AC-05-03: Access My Orders Without Login**
- Given a guest user (not logged in)
- When the user attempts to navigate to `myorder.php`
- Then the user is redirected to `login.php`

---

## AC-06: View Car Tariffs
**Feature:** Displays available car information and rental rates.
**Pages:** `tarrif.php`
**Tables:**

### Acceptance Criteria:

**AC-06-01: Display Car Tariffs Page**
- Given a user (guest or customer) is on the `tarrif.php` page
- When the page loads
- Then a list of available cars with their details (e.g., model, type, rent per day) is displayed

**Complexity: Low — static display page**

---

## Test Data Requirements

*   **Users:**
    *   At least one registered user with valid login credentials (`uname`, `pwd`).
    *   At least one registered user with existing rental orders.
    *   At least one registered user with no rental orders.
*   **Cars:**
    *   A list of available cars with associated rental rates.
    *   A valid `carid` for booking.
    *   An invalid `carid` for testing error handling.
*   **Rental Orders:**
    *   Existing rental orders for a specific user to test the "View My Orders" feature.
*   **Invalid Data:**
    *   Invalid phone numbers.
    *   Invalid email formats.
    *   Invalid dates (past dates, return date before issue date).
    *   Invalid driver's license numbers (if specific format validation exists beyond presence).
    *   Invalid credit card details (if format validation exists beyond presence).