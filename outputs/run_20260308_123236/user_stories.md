# User Story Backlog — Car Rental System

## Epic Summary
*   **User Management:** Manages user accounts, registration, and authentication.
*   **Booking:** Handles the process of renting cars and managing rental requests.
*   **Catalog:** Provides information about available cars and their rental tariffs.

---

## Epic: User Management

### US-001: Register New Customer Account
**As a** Guest
**I want to** fill in my personal details on the `registration.php` page and submit them
**So that** I can create a new customer account in the system.

**Priority:** Must Have
**Story Points:** 5
**Pages:** `registration.php`
**Tables:** `registerhere`

**Acceptance Criteria:**
- [ ] When a guest submits the registration form on `registration.php` with all required fields (`fname`, `lname`, `phone`, `M`, `dob`, `email`, `uname`, `pwd`, `city`, `state`, `country`) filled, a new record is inserted into the `registerhere` table.
- [ ] The system validates the `phone` number using the `validate_phone_number` function before insertion.
- [ ] If the `uname` already exists in the `registerhere` table, an error message is displayed to the user on `registration.php`.
- [ ] Upon successful registration, the user is redirected to `login.php`.

**Notes:** Assumes `M` is a gender selection field.

### US-002: Log in to the System
**As a** Customer
**I want to** enter my `uname` and `pwd` on the `login.php` page and submit them
**So that** I can securely log in to the car rental system.

**Priority:** Must Have
**Story Points:** 3
**Pages:** `login.php`, `session.php`
**Tables:**

**Acceptance Criteria:**
- [ ] When a customer submits their credentials on `login.php`, the system checks if a matching `uname` and `pwd` exist in the `registerhere` table.
- [ ] If credentials match, a user session is created (e.g., `$_SESSION['uname']` is set) and the user is redirected to `rent.php`.
- [ ] If credentials do not match, an error message is displayed on `login.php`.

**Notes:** The actual session handling logic is assumed to be in `session.php`.

### US-003: Log out of the System
**As a** Customer
**I want to** click on a logout option
**So that** my session is terminated and I am logged out of the system.

**Priority:** Must Have
**Story Points:** 1
**Pages:** `logout.php`
**Tables:**

**Acceptance Criteria:**
- [ ] When the logout action is triggered on `logout.php`, all session variables are destroyed.
- [ ] The user is redirected to `login.php`.

**Notes:** This is a simple, essential function.

---

## Epic: Booking

### US-004: Book a Car Rental
**As a** Customer
**I want to** select a car, enter rental details (`doi`, `dor`, `distance`, `dln`, `cc`, `dc`, `no`, `mob`) on `rent.php` and submit the request on `request.php`
**So that** my car rental request is submitted and stored for processing.

**Priority:** Must Have
**Story Points:** 8
**Pages:** `rent.php`, `request.php`
**Tables:** `request`

**Acceptance Criteria:**
- [ ] When a logged-in customer submits the booking form on `request.php`, a new record is inserted into the `request` table with the provided details (`carid`, `doi`, `dor`, `distance`, `dln`, `cc`, `dc`, `no`, `mob`, and the logged-in user's ID).
- [ ] The system validates that `carid` refers to an existing car.
- [ ] The system validates that `doi` (Date of Issue) is a valid date and is not in the past.
- [ ] The system validates that `dor` (Date of Return) is a valid date and is after `doi`.
- [ ] Upon successful submission, the user is redirected back to `rent.php`.

**Notes:** `cc` likely refers to credit card details, `dc` to driver's license, `no` to number of passengers, `mob` to mobile number. `sub` is likely a submit button.

### US-005: View My Rental Orders
**As a** Customer
**I want to** navigate to the `myorder.php` page
**So that** I can view a list of my past and current car rental orders.

**Priority:** Must Have
**Story Points:** 3
**Pages:** `myorder.php`
**Tables:** `request`, `registerhere`

**Acceptance Criteria:**
- [ ] When a logged-in customer accesses `myorder.php`, the system queries the `request` table for all rental requests associated with the logged-in user's ID.
- [ ] The displayed list includes relevant details from the `request` table (e.g., car ID, dates, status).
- [ ] If the user has no orders, a message indicating "No orders found" is displayed.

**Notes:** Assumes user ID is available in the session.

---

## Epic: Catalog

### US-006: View Car Tariffs and Details
**As a** Guest
**I want to** navigate to the `tarrif.php` page
**So that** I can view available cars and their rental rates.

**Priority:** Must Have
**Story Points:** 2
**Pages:** `tarrif.php`
**Tables:**

**Acceptance Criteria:**
- [ ] When a user accesses `tarrif.php`, a list of available cars is displayed.
- [ ] Each car listing includes its details (e.g., model, type) and rental information (e.g., daily rate).

**Notes:** This is a read-only feature.

### US-007: View Car Tariffs and Details (Customer)
**As a** Customer
**I want to** navigate to the `tarrif.php` page
**So that** I can view available cars and their rental rates.

**Priority:** Must Have
**Story Points:** 2
**Pages:** `tarrif.php`
**Tables:**

**Acceptance Criteria:**
- [ ] When a logged-in customer accesses `tarrif.php`, a list of available cars is displayed.
- [ ] Each car listing includes its details (e.g., model, type) and rental information (e.g., daily rate).

**Notes:** This story is identical to US-006 but explicitly covers the Customer role's access.

---

## Backlog Summary Table

| Story ID | Title                               | Epic           | Priority   | Points | Status  |
| :------- | :---------------------------------- | :------------- | :--------- | :----- | :------ |
| US-001   | Register New Customer Account       | User Management| Must Have  | 5      | To Do   |
| US-002   | Log in to the System                | User Management| Must Have  | 3      | To Do   |
| US-003   | Log out of the System               | User Management| Must Have  | 1      | To Do   |
| US-004   | Book a Car Rental                   | Booking        | Must Have  | 8      | To Do   |
| US-005   | View My Rental Orders               | Booking        | Must Have  | 3      | To Do   |
| US-006   | View Car Tariffs and Details (Guest)| Catalog        | Must Have  | 2      | To Do   |
| US-007   | View Car Tariffs and Details (Cust.)| Catalog        | Must Have  | 2      | To Do   |

## Total Story Points

**Must Have:** 24
**Should Have:** 0
**Could Have:** 0
**Won't Have:** 0