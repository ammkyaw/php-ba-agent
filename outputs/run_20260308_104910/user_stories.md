# User Story Backlog — Car Rental System

## Epic Summary
*   **User Management:** Manages user accounts, registration, and login.
*   **Booking and Rental:** Handles car booking, rental requests, and order viewing.
*   **Car Catalog:** Displays car information and rental tariffs.

---

## Epic: User Management

### US-001: Register New User
**As a** Customer
**I want to** register for a new account on the `registration.php` page by providing my first name (`fname`), last name (`lname`), phone number (`phone`), gender (`M`), date of birth (`dob`), email (`email`), username (`uname`), password (`pwd`), city (`city`), state (`state`), and country (`country`)
**So that** I can access the car rental system.

**Priority:** Must Have
**Story Points:** 5
**Pages:** `registration.php`
**Tables:** `registerhere`

**Acceptance Criteria:**
- [ ] Given I am on the `registration.php` page, when I fill in all required fields and click submit, then a new record is inserted into the `registerhere` table with my details.
- [ ] Given I am on the `registration.php` page, when I attempt to register with a username that already exists in the `registerhere` table, then I receive an error message indicating the username is taken.
- [ ] Given I am on the `registration.php` page, when I enter an invalid phone number format, then the `validate_phone_number` function prevents submission and displays an error.
- [ ] Given I have successfully registered, I am redirected to the login page (`login.php`).

**Notes:** The `validate_phone_number` function is assumed to exist and handle phone number format validation.

### US-002: Log In to System
**As a** Customer
**I want to** log in to the system on the `login.php` page using my `uname` and `pwd`
**So that** I can access my account and rent cars.

**Priority:** Must Have
**Story Points:** 3
**Pages:** `login.php`, `session.php`
**Tables:** `registerhere`

**Acceptance Criteria:**
- [ ] Given I am on the `login.php` page, when I enter a valid `uname` and `pwd` that matches a record in the `registerhere` table, then I am redirected to `rent.php` and a user session is created.
- [ ] Given I am on the `login.php` page, when I enter an invalid `uname` or `pwd`, then I remain on the `login.php` page and see an error message.
- [ ] Given I am logged in, a session variable (e.g., `$_SESSION['uname']`) is set to my username.

**Notes:** Assumes `session.php` handles the login processing and session management.

### US-003: Log Out of System
**As a** Customer
**I want to** log out of the system by navigating to `logout.php`
**So that** my session is terminated and I am securely logged out.

**Priority:** Must Have
**Story Points:** 1
**Pages:** `logout.php`
**Tables:**

**Acceptance Criteria:**
- [ ] Given I am logged in, when I navigate to `logout.php`, then my user session is destroyed.
- [ ] Given I have logged out, I am redirected to the `login.php` page.

**Notes:** This is a simple, essential function.

---

## Epic: Booking and Rental

### US-004: Book a Car Rental
**As a** Customer
**I want to** book a car on the `rent.php` page by selecting a `carid`, specifying the date of issue (`doi`), date of return (`dor`), estimated `distance`, driver's license number (`dln`), credit card number (`cc`), credit card expiry date (`dc`), number of `no` of seats, mobile number (`mob`), and submitting the form via `request.php`
**So that** I can reserve a car for my rental period.

**Priority:** Must Have
**Story Points:** 8
**Pages:** `rent.php`, `request.php`
**Tables:** `request`

**Acceptance Criteria:**
- [ ] Given I am logged in and on the `rent.php` page, when I select a car, enter valid `doi`, `dor`, `distance`, `dln`, `cc`, `dc`, `no`, `mob` and click submit, then a new record is inserted into the `request` table with these details.
- [ ] Given I am on the `rent.php` page, when I attempt to submit the booking without selecting a `carid`, then I receive an error message.
- [ ] Given I am on the `rent.php` page, when I attempt to submit the booking with an invalid date format for `doi` or `dor`, then I receive an error message.
- [ ] Given a booking is successful, I am redirected back to the `rent.php` page.

**Notes:** Assumes `carid` is a selectable option on `rent.php`. The system should ideally validate credit card format and expiry date, but this is not explicitly detailed in the model.

### US-005: View My Rental Orders
**As a** Customer
**I want to** view a list of my past and current rental requests on the `myorder.php` page
**So that** I can keep track of my bookings.

**Priority:** Must Have
**Story Points:** 3
**Pages:** `myorder.php`
**Tables:** `request`, `registerhere`

**Acceptance Criteria:**
- [ ] Given I am logged in, when I navigate to `myorder.php`, then I see a list of all rental requests associated with my `uname` from the `request` table.
- [ ] Given I am logged in, when I navigate to `myorder.php`, then the displayed orders include relevant details such as `carid`, `doi`, `dor`, and total cost (if calculable from `distance`, `no`, `mob`, `sub`).
- [ ] Given I am not logged in, when I attempt to access `myorder.php`, then I am redirected to the `login.php` page.

**Notes:** Assumes `myorder.php` queries the `request` table and joins with `registerhere` (or uses session data) to filter by the logged-in user.

---

## Epic: Car Catalog

### US-006: View Car Tariffs
**As a** Customer
**I want to** view available car information and rental rates on the `tarrif.php` page
**So that** I can make informed decisions about which car to rent.

**Priority:** Must Have
**Story Points:** 2
**Pages:** `tarrif.php`
**Tables:**

**Acceptance Criteria:**
- [ ] Given I navigate to the `tarrif.php` page, then a list of available cars is displayed.
- [ ] Given the list of cars is displayed, then each car shows its relevant details and pricing information.

**Notes:** This is a static display page based on available car data, which is not detailed in the provided model.

---

## Backlog Summary Table

| Story ID | Title                 | Epic             | Priority | Points | Status  |
| :------- | :-------------------- | :--------------- | :------- | :----- | :------ |
| US-001   | Register New User     | User Management  | Must Have| 5      | To Do   |
| US-002   | Log In to System      | User Management  | Must Have| 3      | To Do   |
| US-003   | Log Out of System     | User Management  | Must Have| 1      | To Do   |
| US-004   | Book a Car Rental     | Booking and Rental | Must Have| 8      | To Do   |
| US-005   | View My Rental Orders | Booking and Rental | Must Have| 3      | To Do   |
| US-006   | View Car Tariffs      | Car Catalog      | Must Have| 2      | To Do   |

## Total Story Points

**Must Have:** 22
**Should Have:** 0
**Could Have:** 0
**Won't Have:** 0