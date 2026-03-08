# Acceptance Criteria — Car Rental System

## Overview
Acceptance testing for the Car Rental System will focus on verifying the functionality described in the domain model. Each feature will be tested according to its defined pages, tables, inputs, and business rules. Testing will include happy path scenarios, validation failures, and edge cases where applicable.

---

## AC-01: User Registration
**Feature:** Allows new users to create an account in the system.
**Pages:** `registration.php`
**Tables:** `registerhere`

### Acceptance Criteria:

**AC-01-01: Successful User Registration**
- Given the user is on the `registration.php` page
- When the user fills in all required fields with valid data (including unique `uname` and correctly formatted `phone`) and submits the form
- Then a new row is inserted into the `registerhere` table with the provided user details.
- And the user is redirected to a confirmation or login page.

**AC-01-02: Username Already Exists**
- Given the user is on the `registration.php` page
- When the user attempts to register with a `uname` that already exists in the `registerhere` table
- Then an error message indicating that the username is already taken is displayed.
- And no new row is inserted into the `registerhere` table.

**AC-01-03: Invalid Phone Number Format**
- Given the user is on the `registration.php` page
- When the user enters an invalid format for the `phone` field and submits the form
- Then an error message indicating an invalid phone number format is displayed.
- And no new row is inserted into the `registerhere` table.

**AC-01-04: Missing Required Field**
- Given the user is on the `registration.php` page
- When the user submits the registration form with a required field (e.g., `email`, `pwd`) left empty
- Then an error message indicating the missing required field is displayed.
- And no new row is inserted into the `registerhere` table.

---

## AC-02: User Login
**Feature:** Allows existing users to log in to the system.
**Pages:** `login.php`, `session.php`
**Tables:** `registerhere`

### Acceptance Criteria:

**AC-02-01: Successful User Login**
- Given the user is on the `login.php` page
- When the user enters a valid `uname` and `pwd` that match a record in the `registerhere` table and submits the form
- Then the user's session is created (e.g., `$_SESSION["user_id"]` is set).
- And the user is redirected to `rent.php`.

**AC-02-02: Invalid Credentials**
- Given the user is on the `login.php` page
- When the user enters an invalid `uname` or `pwd` that does not match any record in the `registerhere` table and submits the form
- Then an error message indicating invalid credentials is displayed.
- And the user remains on the `login.php` page.

**AC-02-03: Accessing Protected Page Without Login**
- Given the user's session is not set (`$_SESSION["user_id"]` is not set)
- When the user attempts to access `rent.php` directly
- Then the user is redirected to `login.php` via `header()`.

---

## AC-03: Car Rental Booking
**Feature:** Enables authenticated users to select a car and book it for a specified period.
**Pages:** `rent.php`, `request.php`
**Tables:** `request`

### Acceptance Criteria:

**AC-03-01: Successful Car Rental Booking**
- Given the user is authenticated and on the `rent.php` page
- When the user selects a `carid`, enters valid `doi`, `dor`, `distance`, `dln`, `cc`, `dc`, `no`, `mob`, `sub` and submits the booking form
- Then a new row is inserted into the `request` table with the booking details.
- And the user is redirected back to `rent.php`.

**AC-03-02: Booking Without Selecting a Car**
- Given the user is authenticated and on the `rent.php` page
- When the user attempts to submit the booking form without selecting a `carid`
- Then an error message indicating that a car must be selected is displayed.
- And no row is inserted into the `request` table.

**AC-03-03: Booking With Invalid Date Range**
- Given the user is authenticated and on the `rent.php` page
- When the user enters an `doi` that is after `dor` and submits the booking form
- Then an error message indicating an invalid date range is displayed.
- And no row is inserted into the `request` table.

**AC-03-04: Booking With Missing Payment Information**
- Given the user is authenticated and on the `rent.php` page
- When the user attempts to submit the booking form without providing required payment details (`cc`, `dc`)
- Then an error message indicating missing payment information is displayed.
- And no row is inserted into the `request` table.

---

## AC-04: View My Orders
**Feature:** Allows authenticated users to view their past and current rental requests.
**Pages:** `myorder.php`
**Tables:** `request`, `registerhere`

### Acceptance Criteria:

**AC-04-01: Display User's Rental Orders**
- Given the user is authenticated and has previously made rental requests
- When the user navigates to `myorder.php`
- Then a list of the user's rental requests from the `request` table is displayed, associated with their `user_id`.

**AC-04-02: Display Message for No Orders**
- Given the user is authenticated and has no previous rental requests
- When the user navigates to `myorder.php`
- Then a message indicating that there are no rental orders is displayed.

**AC-04-03: Accessing My Orders Without Login**
- Given the user's session is not set (`$_SESSION["user_id"]` is not set)
- When the user attempts to access `myorder.php` directly
- Then the user is redirected to `login.php` via `header()`.

---

## AC-05: User Logout
**Feature:** Allows authenticated users to log out of the system.
**Pages:** `logout.php`
**Tables:** None

### Acceptance Criteria:

**AC-05-01: Successful User Logout**
- Given the user is authenticated
- When the user clicks on the logout link/button (which navigates to `logout.php`)
- Then the user's session is terminated.
- And the user is redirected to `login.php`.

**Complexity: Low — single operation**

---

## AC-06: View Car Tariffs
**Feature:** Displays available car information and rental rates to users.
**Pages:** `tarrif.php`
**Tables:** None

### Acceptance Criteria:

**AC-06-01: Display Car Tariff Information**
- Given the user is on the `tarrif.php` page
- When the page loads
- Then a list of available cars with their details and pricing is displayed.

**Complexity: Low — single operation**

---

## Test Data Requirements
*   **Users:**
    *   At least two registered users with valid credentials for login testing.
    *   One registered user with a username that already exists for testing duplicate registration.
    *   One user with an invalid phone number format for registration testing.
*   **Cars:**
    *   A list of cars with associated tariff information to be displayed on `tarrif.php`.
*   **Rental Requests:**
    *   Several rental requests associated with at least one registered user to test the "View My Orders" feature.
    *   No rental requests for a user to test the "no orders" message.
*   **Booking Data:**
    *   Valid `carid`, `doi`, `dor`, `distance`, `dln`, `cc`, `dc`, `no`, `mob`, `sub` for successful booking.
    *   Invalid `doi` (e.g., `doi` after `dor`) for testing date range validation.
    *   Missing required fields for booking (e.g., `carid`, `cc`).