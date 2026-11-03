/**
 * MessMind AI - Smart Mess Node.js Backend Server
 * High-performance Express backend connected to SQLite (data/messmind.db)
 * Supports 1,200-student capacity, real-time connectivity, authentication,
 * Student Portal, Admin Portal, Planner, Waste Logging, and Analytics.
 */

const express = require('express');
const cors = require('cors');
const path = require('node:path');
const fs = require('node:fs');
const crypto = require('node:crypto');
const { DatabaseSync } = require('node:sqlite');

const app = express();
const PORT = process.env.PORT || 3000;
let DB_PATH = path.join(__dirname, 'data', 'messmind.db');
const MENU_PATH = path.join(__dirname, 'data', 'weekly_menu.json');

// Vercel serverless environment compatibility (writable /tmp)
if (process.env.VERCEL) {
    const tmpDbPath = path.join('/tmp', 'messmind.db');
    if (!fs.existsSync(tmpDbPath)) {
        try {
            fs.copyFileSync(DB_PATH, tmpDbPath);
        } catch (e) {
            console.error('Failed to copy SQLite db to /tmp on Vercel:', e);
        }
    }
    DB_PATH = tmpDbPath;
}

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// SQLite Connection
const db = new DatabaseSync(DB_PATH);

// Safe schema migration for users table
function initServerDb() {
    try {
        const info = db.prepare("PRAGMA table_info(users)").all();
        const cols = info.map(c => c.name);
        if (!cols.includes('student_roll_number')) {
            db.prepare("ALTER TABLE users ADD COLUMN student_roll_number TEXT DEFAULT NULL").run();
        }
        if (!cols.includes('account_status')) {
            db.prepare("ALTER TABLE users ADD COLUMN account_status TEXT NOT NULL DEFAULT 'Active'").run();
        }
        db.prepare("UPDATE users SET student_roll_number = UPPER(username) WHERE role = 'Student' AND (student_roll_number IS NULL OR student_roll_number = '')").run();
        db.prepare("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)").run();
        db.prepare("CREATE INDEX IF NOT EXISTS idx_users_roll ON users(student_roll_number)").run();
        db.prepare("CREATE INDEX IF NOT EXISTS idx_users_role_status ON users(role, account_status)").run();

        // System Settings Table (Active Menu Week)
        db.prepare(`
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        `).run();
        db.prepare("INSERT OR IGNORE INTO system_settings (key, value) VALUES ('active_menu_week', 'Week A')").run();
    } catch (e) {
        console.error("Database schema init error:", e.message);
    }
}
initServerDb();

// Helper: Secure password hashing (100,000 PBKDF2-HMAC-SHA256 iterations)
function hashPassword(password, salt) {
    const activeSalt = salt || crypto.randomBytes(16).toString('hex');
    const hash = crypto.pbkdf2Sync(password, activeSalt, 100000, 32, 'sha256').toString('hex');
    return { hash, salt: activeSalt };
}

// Helper: Password verification with constant-time equality check
function verifyPassword(providedPassword, storedHash, salt) {
    if (!storedHash || !salt || !providedPassword) return false;
    try {
        const hash = crypto.pbkdf2Sync(providedPassword, salt, 100000, 32, 'sha256').toString('hex');
        return crypto.timingSafeEqual(Buffer.from(hash, 'utf8'), Buffer.from(storedHash, 'utf8'));
    } catch (e) {
        return false;
    }
}

// Load Menu
let weeklyMenuData = null;
function loadMenu() {
    try {
        const raw = fs.readFileSync(MENU_PATH, 'utf8');
        weeklyMenuData = JSON.parse(raw);
    } catch (err) {
        console.error('Error loading weekly menu JSON:', err);
    }
}
loadMenu();

// Helper: Resolve dish for Week A vs Week B
function resolveDishName(dishStr, weekChoice = 'Week A') {
    if (!dishStr) return '';
    if (dishStr.includes(' / ')) {
        const parts = dishStr.split(' / ').map(p => p.trim());
        return weekChoice === 'Week B' ? (parts[1] || parts[0]) : parts[0];
    }
    if (dishStr.includes('/')) {
        const parts = dishStr.split('/').map(p => p.trim());
        return weekChoice === 'Week B' ? (parts[1] || parts[0]) : parts[0];
    }
    return dishStr.trim();
}

// Helper: Portion specs matching reference planner
function getDishPortionSpecs(dishName) {
    const dLower = dishName.toLowerCase();
    if (dLower.includes('rice') || dLower.includes('pulao') || dLower.includes('pulav') || dLower.includes('khichadi') || dLower.includes('pongal')) {
        return { category: 'Rice/grain', perHeadVal: 100, unit: 'g', display: '100 g' };
    }
    if (dLower.includes('dal') || dLower.includes('sambar') || dLower.includes('rasam') || dLower.includes('curry') || dLower.includes('gravy')) {
        return { category: 'Dal/gravy', perHeadVal: 30, unit: 'g', display: '30 g' };
    }
    if (dLower.includes('chapati') || dLower.includes('fulka') || dLower.includes('roti') || dLower.includes('puri') || dLower.includes('paratha')) {
        return { category: 'Bread', perHeadVal: 3, unit: 'pc', display: '3 pc' };
    }
    if (dLower.includes('buttermilk') || dLower.includes('tea') || dLower.includes('coffee') || dLower.includes('milk') || dLower.includes('lemonade')) {
        return { category: 'Drink', perHeadVal: 150, unit: 'ml', display: '150 ml' };
    }
    if (dLower.includes('sweet') || dLower.includes('halwa') || dLower.includes('sheera') || dLower.includes('jalebi') || dLower.includes('gulab')) {
        return { category: 'Sweet', perHeadVal: 50, unit: 'g', display: '50 g' };
    }
    // Default Sabzi / side
    return { category: 'Sabzi', perHeadVal: 100, unit: 'g', display: '100 g' };
}

// Attendance fallback moving average rates (for 1,200 capacity)
const HISTORICAL_ATTENDANCE_RATES = {
    Breakfast: 0.78,
    Lunch: 0.92,
    Snacks: 0.70,
    Dinner: 0.88
};

const BASELINE_REPLY_ESTIMATES = {
    Breakfast: { replied: 1047, eating: 790 },
    Lunch: { replied: 684, eating: 644 },
    Snacks: { replied: 894, eating: 606 },
    Dinner: { replied: 817, eating: 721 }
};

// Helper: Calculate live headcount for date + meal
function calculateHeadcount(mealDate, mealType, dataSource = 'ALL') {
    const rate = HISTORICAL_ATTENDANCE_RATES[mealType] || 0.85;
    const base = BASELINE_REPLY_ESTIMATES[mealType] || { replied: 800, eating: 680 };

    let whereCond = 'meal_date = ? AND meal_type = ?';
    const params = [mealDate, mealType];

    if (dataSource === 'REAL') {
        whereCond += " AND data_source = 'REAL'";
    } else if (dataSource === 'SYNTHETIC') {
        whereCond += " AND data_source = 'SYNTHETIC'";
    }

    const stmt = db.prepare(`
        SELECT attendance_intent, COUNT(*) as cnt
        FROM student_registrations
        WHERE ${whereCond}
        GROUP BY attendance_intent
    `);
    const rows = stmt.all(...params);

    let dbEating = 0;
    let dbSkipping = 0;
    for (const r of rows) {
        if (r.attendance_intent === 'Yes') dbEating += Number(r.cnt);
        if (r.attendance_intent === 'No') dbSkipping += Number(r.cnt);
    }
    const dbTotal = dbEating + dbSkipping;

    let replied = 0;
    let eating = 0;
    let skipping = 0;

    if (dataSource === 'REAL') {
        replied = Math.min(1200, dbTotal);
        eating = Math.min(replied, dbEating);
        skipping = dbSkipping;
    } else {
        replied = base.replied;
        eating = base.eating;
        if (dbTotal > 0) {
            replied = Math.min(1200, Math.max(dbTotal, base.replied));
            eating = Math.min(replied, Math.max(dbEating, base.eating));
        }
        skipping = Math.max(0, replied - eating);
    }

    const pending = Math.max(0, 1200 - replied);
    const estimatedFromPending = Math.round(pending * rate);
    const predictedHeadcount = eating + estimatedFromPending;
    const fewerPlates = Math.max(0, 1200 - predictedHeadcount);

    return {
        meal_date: mealDate,
        meal_type: mealType,
        total_capacity: 1200,
        replied,
        eating,
        skipping,
        pending,
        rate,
        rate_pct: Math.round(rate * 100),
        estimated_from_pending: estimatedFromPending,
        predicted_headcount: predictedHeadcount,
        fewer_plates: fewerPlates,
        has_real_data: dbTotal > 0,
        insufficient_data: (dataSource === 'REAL' && dbTotal === 0)
    };
}

// -----------------------------------------------------------------------------
// AUTHENTICATION ROUTES
// -----------------------------------------------------------------------------
app.post('/api/auth/login', (req, res) => {
    const { username, password } = req.body;
    if (!username || !password) {
        return res.status(400).json({ success: false, message: 'Username and password are required.' });
    }

    const cleanInput = username.trim();
    const cleanUserLower = cleanInput.toLowerCase();
    const cleanUserUpper = cleanInput.toUpperCase();

    const stmt = db.prepare(`
        SELECT user_id, username, student_roll_number, password_hash, salt, role, full_name, account_status
        FROM users
        WHERE (LOWER(username) = ? OR UPPER(student_roll_number) = ? OR UPPER(username) = ?)
          AND account_status = 'Active'
    `);
    const user = stmt.get(cleanUserLower, cleanUserUpper, cleanUserUpper);

    if (!user) {
        return res.status(401).json({ success: false, message: 'Invalid credentials. User not found or account inactive.' });
    }

    let isValid = verifyPassword(password, user.password_hash, user.salt);

    // Support initial credentials (roll number or Student@123 / Admin@123) and update hash
    if (!isValid) {
        if (user.role === 'Student' && (password.toUpperCase() === cleanUserUpper || password === 'Student@123')) {
            isValid = true;
            const { hash: newHash, salt: newSalt } = hashPassword(password);
            db.prepare('UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?').run(newHash, newSalt, user.user_id);
        } else if (user.role === 'Admin' && password === 'Admin@123') {
            isValid = true;
        }
    }

    if (isValid) {
        const roll = user.student_roll_number || (user.role === 'Student' ? user.username.toUpperCase() : null);
        return res.json({
            success: true,
            user: {
                user_id: user.user_id,
                username: user.role === 'Student' ? user.username.toUpperCase() : user.username,
                student_roll_number: roll,
                role: user.role,
                full_name: user.full_name,
                account_status: user.account_status
            }
        });
    }

    return res.status(401).json({ success: false, message: 'Incorrect password.' });
});

// Admin: Create Student or Staff Account
app.post('/api/admin/users/create', (req, res) => {
    const { username, password, role, full_name, student_roll_number } = req.body;
    if (!username || !password || !role) {
        return res.status(400).json({ success: false, message: 'Username, password, and role are required.' });
    }

    const cleanRole = role.trim();
    if (!['Admin', 'Kitchen Staff', 'Student'].includes(cleanRole)) {
        return res.status(400).json({ success: false, message: 'Role must be Admin, Kitchen Staff, or Student.' });
    }
    if (password.length < 6) {
        return res.status(400).json({ success: false, message: 'Password must be at least 6 characters long.' });
    }

    const cleanUname = username.trim().toLowerCase();
    const roll = cleanRole === 'Student' ? (student_roll_number || cleanUname).trim().toUpperCase() : null;

    // Check duplicate username or roll number
    const dupStmt = db.prepare('SELECT user_id FROM users WHERE LOWER(username) = ? OR (student_roll_number IS NOT NULL AND UPPER(student_roll_number) = ?)');
    const dup = dupStmt.get(cleanUname, roll || '');
    if (dup) {
        return res.status(409).json({ success: false, message: `Account with username '${cleanUname}' or roll number '${roll}' already exists.` });
    }

    const { hash: pwdHash, salt } = hashPassword(password);
    const nowStr = new Date().toISOString();
    const name = (full_name || `${cleanRole} ${cleanUname.toUpperCase()}`).trim();

    const insertStmt = db.prepare(`
        INSERT INTO users (username, student_roll_number, password_hash, salt, role, full_name, created_at, account_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'Active')
    `);
    const info = insertStmt.run(cleanUname, roll, pwdHash, salt, cleanRole, name, nowStr);

    return res.status(201).json({
        success: true,
        message: `Account for ${cleanUname.toUpperCase()} created successfully with role ${cleanRole}.`,
        user: {
            user_id: Number(info.lastInsertRowid),
            username: cleanRole === 'Student' ? cleanUname.toUpperCase() : cleanUname,
            student_roll_number: roll,
            role: cleanRole,
            full_name: name,
            account_status: 'Active'
        }
    });
});

// Active Menu Week Config
app.get('/api/config/active-week', (req, res) => {
    try {
        const row = db.prepare("SELECT value FROM system_settings WHERE key = 'active_menu_week'").get();
        return res.json({ success: true, active_week: row?.value || 'Week A' });
    } catch (e) {
        return res.json({ success: true, active_week: 'Week A' });
    }
});

app.post('/api/admin/active-week', (req, res) => {
    const { week } = req.body;
    const cleanWeek = (week && week.toLowerCase().includes('b')) ? 'Week B' : 'Week A';
    try {
        db.prepare(`
            INSERT INTO system_settings (key, value) VALUES ('active_menu_week', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        `).run(cleanWeek);
        return res.json({ success: true, active_week: cleanWeek, message: `Official active menu week set to ${cleanWeek}.` });
    } catch (e) {
        return res.status(500).json({ success: false, message: e.message });
    }
});

// Admin: User Counts & Statistics
app.get('/api/admin/users/stats', (req, res) => {
    try {
        const row = db.prepare(`
            SELECT 
                COUNT(*) as total_users,
                SUM(CASE WHEN role = 'Student' THEN 1 ELSE 0 END) as total_students,
                SUM(CASE WHEN role = 'Kitchen Staff' THEN 1 ELSE 0 END) as total_staff,
                SUM(CASE WHEN role = 'Admin' THEN 1 ELSE 0 END) as total_admins,
                SUM(CASE WHEN account_status = 'Active' THEN 1 ELSE 0 END) as active_accounts,
                SUM(CASE WHEN account_status = 'Inactive' THEN 1 ELSE 0 END) as inactive_accounts
            FROM users
        `).get();
        return res.json({
            success: true,
            total_users: Number(row.total_users || 0),
            total_students: Number(row.total_students || 0),
            total_staff: Number(row.total_staff || 0),
            total_admins: Number(row.total_admins || 0),
            active_accounts: Number(row.active_accounts || 0),
            inactive_accounts: Number(row.inactive_accounts || 0)
        });
    } catch (e) {
        return res.status(500).json({ success: false, message: e.message });
    }
});

// Admin: Filtered & Searchable User Directory
app.get('/api/admin/users', (req, res) => {
    const search = (req.query.search || '').trim().toLowerCase();
    const role = req.query.role || 'All';
    const status = req.query.status || 'All';
    const limit = Math.min(100, Math.max(10, parseInt(req.query.limit) || 50));
    const offset = Math.max(0, parseInt(req.query.offset) || 0);

    const conds = [];
    const params = [];

    if (search) {
        conds.push('(LOWER(username) LIKE ? OR LOWER(IFNULL(student_roll_number, "")) LIKE ? OR LOWER(full_name) LIKE ?)');
        const s = `%${search}%`;
        params.push(s, s, s);
    }
    if (role && role !== 'All') {
        conds.push('role = ?');
        params.push(role);
    }
    if (status && status !== 'All') {
        conds.push('account_status = ?');
        params.push(status);
    }

    const whereClause = conds.length > 0 ? `WHERE ${conds.join(' AND ')}` : '';

    const countRow = db.prepare(`SELECT COUNT(*) as cnt FROM users ${whereClause}`).get(...params);
    const totalMatching = Number(countRow?.cnt || 0);

    const stmt = db.prepare(`
        SELECT user_id, username, student_roll_number, role, full_name, created_at, account_status
        FROM users
        ${whereClause}
        ORDER BY user_id DESC
        LIMIT ? OFFSET ?
    `);
    const users = stmt.all(...params, limit, offset);

    return res.json({ success: true, count: users.length, total: totalMatching, users });
});

// Admin: Edit User Details
app.post('/api/admin/users/update', (req, res) => {
    const { user_id, full_name, student_roll_number, role } = req.body;
    const uid = parseInt(user_id);
    if (!uid || !full_name) {
        return res.status(400).json({ success: false, message: 'User ID and full name are required.' });
    }

    const existing = db.prepare('SELECT user_id, username, role FROM users WHERE user_id = ?').get(uid);
    if (!existing) {
        return res.status(404).json({ success: false, message: 'User not found.' });
    }

    const newRole = role || existing.role;
    let roll = null;
    if (newRole === 'Student') {
        roll = (student_roll_number || existing.username).trim().toUpperCase();
        const dup = db.prepare('SELECT user_id FROM users WHERE UPPER(student_roll_number) = ? AND user_id != ?').get(roll, uid);
        if (dup) {
            return res.status(409).json({ success: false, message: `Another student already has roll number '${roll}'.` });
        }
    }

    db.prepare(`
        UPDATE users
        SET full_name = ?, student_roll_number = ?, role = ?
        WHERE user_id = ?
    `).run(full_name.trim(), roll, newRole, uid);

    return res.json({ success: true, message: `User ${existing.username.toUpperCase()} updated successfully.` });
});

// Admin: Toggle Account Status (Active / Inactive)
app.post('/api/admin/users/toggle-status', (req, res) => {
    const { user_id, account_status } = req.body;
    const uid = parseInt(user_id);
    if (!uid) {
        return res.status(400).json({ success: false, message: 'User ID is required.' });
    }

    const existing = db.prepare('SELECT user_id, username, role FROM users WHERE user_id = ?').get(uid);
    if (!existing) {
        return res.status(404).json({ success: false, message: 'User not found.' });
    }

    const cleanStatus = (account_status && account_status.toLowerCase() === 'inactive') ? 'Inactive' : 'Active';
    if (existing.username.toLowerCase() === 'admin' && cleanStatus === 'Inactive') {
        return res.status(400).json({ success: false, message: 'Cannot deactivate primary system administrator account.' });
    }

    db.prepare('UPDATE users SET account_status = ? WHERE user_id = ?').run(cleanStatus, uid);
    return res.json({ success: true, message: `Account '${existing.username}' set to ${cleanStatus}.` });
});

// Admin: Reset Password Securely
app.post('/api/admin/users/reset-password', (req, res) => {
    const { user_id, new_password } = req.body;
    const uid = parseInt(user_id);
    if (!uid || !new_password || new_password.length < 6) {
        return res.status(400).json({ success: false, message: 'Password must be at least 6 characters long.' });
    }

    const existing = db.prepare('SELECT user_id, username FROM users WHERE user_id = ?').get(uid);
    if (!existing) {
        return res.status(404).json({ success: false, message: 'User not found.' });
    }

    const { hash: newHash, salt: newSalt } = hashPassword(new_password);
    db.prepare('UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?').run(newHash, newSalt, uid);

    return res.json({ success: true, message: `Password for '${existing.username}' reset successfully.` });
});

// -----------------------------------------------------------------------------
// MENU & DISH RESOLUTION
// -----------------------------------------------------------------------------
app.get('/api/menu', (req, res) => {
    const week = req.query.week || 'Week A';
    if (!weeklyMenuData) loadMenu();
    return res.json({
        metadata: weeklyMenuData.metadata,
        menu: weeklyMenuData.menu,
        week
    });
});

// -----------------------------------------------------------------------------
// STUDENT PORTAL APIS
// -----------------------------------------------------------------------------

// Get Student Dashboard Status
app.get('/api/student/dashboard', (req, res) => {
    const roll = (req.query.roll || '24BDS021').toUpperCase();
    const dateStr = req.query.date || new Date().toISOString().split('T')[0];

    // Attendance choices for this date
    const attStmt = db.prepare(`
        SELECT meal_type, attendance_intent
        FROM student_registrations
        WHERE UPPER(registration_code) = ? AND meal_date = ?
    `);
    const attRows = attStmt.all(roll, dateStr);
    const mealStatus = {};
    for (const r of attRows) {
        mealStatus[r.meal_type] = r.attendance_intent; // 'Yes' or 'No'
    }

    // Impact Calculation
    // Total skipped meals
    const skipStmt = db.prepare(`
        SELECT COUNT(*) as cnt
        FROM student_registrations
        WHERE UPPER(registration_code) = ? AND attendance_intent = 'No'
    `);
    const skipCount = Number(skipStmt.get(roll)?.cnt || 0);

    // Total check-ins
    const totalStmt = db.prepare(`
        SELECT COUNT(*) as cnt
        FROM student_registrations
        WHERE UPPER(registration_code) = ?
    `);
    const totalEntries = Number(totalStmt.get(roll)?.cnt || 0);

    // Total ratings
    const ratingStmt = db.prepare(`
        SELECT COUNT(*) as cnt
        FROM dish_likes
        WHERE UPPER(registration_code) = ?
    `);
    const ratingCount = Number(ratingStmt.get(roll)?.cnt || 0);

    const greenPoints = (totalEntries * 10) + (ratingCount * 5);
    let level = 'Seedling';
    if (greenPoints >= 100) level = 'Eco-Warrior';
    else if (greenPoints >= 50) level = 'Sprout';

    const foodAvoidedKg = (skipCount * 0.35).toFixed(2);

    // Unread notifications count
    const notifStmt = db.prepare(`
        SELECT id FROM system_alerts WHERE target_role IN ('Student', 'All')
    `);
    const allAlerts = notifStmt.all();

    const readStmt = db.prepare(`
        SELECT alert_id FROM alert_read_receipts WHERE UPPER(user_id_or_roll) = ?
    `);
    const readAlertIds = new Set(readStmt.all(roll).map(r => r.alert_id));
    const unreadCount = allAlerts.filter(a => !readAlertIds.has(a.id)).length;

    return res.json({
        roll,
        date: dateStr,
        mealStatus,
        impact: {
            greenPoints,
            level,
            mealsSkipped: skipCount,
            foodAvoidedKg
        },
        unreadNotifications: unreadCount
    });
});

// Record Student Attendance (Eating / Skipping)
app.post('/api/student/attendance', (req, res) => {
    const { roll, date: mealDate, meal, intent } = req.body;
    if (!roll || !mealDate || !meal || !intent) {
        return res.status(400).json({ success: false, message: 'Missing required parameters.' });
    }

    const cleanRoll = roll.trim().toUpperCase();
    const cleanIntent = intent === 'Yes' ? 'Yes' : 'No';
    const nowStr = new Date().toISOString();

    const stmt = db.prepare(`
        INSERT OR REPLACE INTO student_registrations (
            meal_date, meal_type, selected_dishes, registration_code,
            intention_to_attend, attendance_intent, data_source, submission_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, 'REAL', ?)
    `);

    stmt.run(mealDate, meal, JSON.stringify([]), cleanRoll, cleanIntent === 'Yes' ? 1 : 0, cleanIntent, nowStr);

    return res.json({
        success: true,
        message: `Attendance updated to '${cleanIntent === 'Yes' ? 'Eating' : 'Skipping'}' for ${meal}.`,
        meal,
        intent: cleanIntent
    });
});

// Record Dish Rating (👍 = 1, 👎 = 0)
app.post('/api/student/rating', (req, res) => {
    const { roll, dish, meal, date: mealDate, liked } = req.body;
    if (!dish || !meal || liked === undefined) {
        return res.status(400).json({ success: false, message: 'Missing required parameters.' });
    }

    const cleanRoll = (roll || '24BDS021').trim().toUpperCase();
    const cleanLiked = Number(liked) === 1 ? 1 : 0;
    const nowStr = new Date().toISOString();
    const cleanDate = mealDate || new Date().toISOString().split('T')[0];

    const stmt = db.prepare(`
        INSERT INTO dish_likes (dish_name, meal_type, meal_date, registration_code, liked, submitted_timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    `);
    stmt.run(dish, meal, cleanDate, cleanRoll, cleanLiked, nowStr);

    // Calculate updated like %
    const aggStmt = db.prepare(`
        SELECT
            SUM(CASE WHEN liked = 1 THEN 1 ELSE 0 END) as likes,
            COUNT(*) as total
        FROM dish_likes
        WHERE dish_name = ?
    `);
    const agg = aggStmt.get(dish);
    const pct = agg.total > 0 ? Math.round((agg.likes / agg.total) * 100) : 50;

    return res.json({
        success: true,
        dish,
        liked: cleanLiked,
        liked_pct: pct,
        total_votes: agg.total
    });
});

// Get Student Notifications
app.get('/api/student/notifications', (req, res) => {
    const roll = (req.query.roll || '24BDS021').toUpperCase();

    // Fetch alerts
    const alertsStmt = db.prepare(`
        SELECT id, created_at, alert_type, title, message, severity, target_role, metadata_json
        FROM system_alerts
        WHERE target_role IN ('Student', 'All')
        ORDER BY id DESC
        LIMIT 25
    `);
    const alerts = alertsStmt.all();

    // Fetch read receipts
    const readStmt = db.prepare(`
        SELECT alert_id FROM alert_read_receipts WHERE UPPER(user_id_or_roll) = ?
    `);
    const readIds = new Set(readStmt.all(roll).map(r => r.alert_id));

    const formatted = alerts.map(a => {
        let meta = {};
        if (a.metadata_json) {
            try { meta = JSON.parse(a.metadata_json); } catch (e) {}
        }
        return {
            id: a.id,
            title: a.title,
            message: a.message,
            severity: a.severity,
            created_at: a.created_at,
            is_read: readIds.has(a.id),
            metadata: meta
        };
    });

    return res.json({
        roll,
        notifications: formatted,
        unreadCount: formatted.filter(n => !n.is_read).length
    });
});

// Mark Notification Read
app.post('/api/student/notifications/read', (req, res) => {
    const { roll, alert_id, all } = req.body;
    const cleanRoll = (roll || '24BDS021').toUpperCase();
    const nowStr = new Date().toISOString();

    if (all) {
        const notifStmt = db.prepare("SELECT id FROM system_alerts WHERE target_role IN ('Student', 'All')");
        const alerts = notifStmt.all();
        const insertStmt = db.prepare(`
            INSERT OR IGNORE INTO alert_read_receipts (alert_id, user_id_or_roll, read_at)
            VALUES (?, ?, ?)
        `);
        for (const a of alerts) {
            insertStmt.run(a.id, cleanRoll, nowStr);
        }
        return res.json({ success: true, message: 'All notifications marked as read.' });
    }

    if (alert_id) {
        const insertStmt = db.prepare(`
            INSERT OR IGNORE INTO alert_read_receipts (alert_id, user_id_or_roll, read_at)
            VALUES (?, ?, ?)
        `);
        insertStmt.run(Number(alert_id), cleanRoll, nowStr);
        return res.json({ success: true, message: 'Notification marked as read.' });
    }

    return res.status(400).json({ success: false, message: 'Missing parameters.' });
});

// -----------------------------------------------------------------------------
// ADMIN PORTAL APIS
// -----------------------------------------------------------------------------

// Admin Dashboard KPIs and Headcounts
app.get('/api/admin/dashboard', (req, res) => {
    const targetDate = req.query.date || new Date().toISOString().split('T')[0];
    const costPerKg = Number(req.query.cost_per_kg) || 35.0;
    const dataSource = (req.query.data_source || 'ALL').toUpperCase();

    // Headcounts for all 4 meals
    const meals = ['Breakfast', 'Lunch', 'Snacks', 'Dinner'];
    const headcountData = {};
    let totalAvoidedPlates = 0;
    let anyRealData = false;

    for (const m of meals) {
        const hc = calculateHeadcount(targetDate, m, dataSource);
        headcountData[m] = hc;
        totalAvoidedPlates += hc.fewer_plates;
        if (hc.has_real_data) anyRealData = true;
    }

    // Waste logs aggregation
    let wasteQuery = 'SELECT COALESCE(SUM(weight_kg), 0.0) as live_waste, COUNT(*) as cnt FROM waste_logs';
    let destQuery = 'SELECT destination, COALESCE(SUM(weight_kg), 0.0) as wt FROM waste_logs';
    if (dataSource === 'REAL') {
        wasteQuery += " WHERE data_source = 'REAL'";
        destQuery += " WHERE data_source = 'REAL'";
    } else if (dataSource === 'SYNTHETIC') {
        wasteQuery += " WHERE data_source = 'SYNTHETIC'";
        destQuery += " WHERE data_source = 'SYNTHETIC'";
    }
    destQuery += ' GROUP BY destination';

    const wasteRow = db.prepare(wasteQuery).get();
    const liveWasteKg = Number(wasteRow?.live_waste || 0);
    const wasteLogCount = Number(wasteRow?.cnt || 0);

    const destRows = db.prepare(destQuery).all();
    const liveByDest = {};
    for (const r of destRows) {
        liveByDest[r.destination] = Number(r.wt);
    }

    const baselineWeekKg = 460.0;
    let totalWasteKg, totalCookedKg, totalEatenKg, ngoKg, compKg, discKg;
    const insufficientData = (dataSource === 'REAL' && (!anyRealData || wasteLogCount === 0));

    if (dataSource === 'REAL') {
        totalWasteKg = Math.round(liveWasteKg * 10) / 10;
        totalCookedKg = Math.round((totalWasteKg + (anyRealData ? 3392.0 : 0)) * 10) / 10;
        totalEatenKg = Math.max(0, Math.round((totalCookedKg - totalWasteKg) * 10) / 10);
        ngoKg = Math.round((liveByDest['NGO'] || 0) + (liveByDest['NGO Donation'] || 0));
        compKg = Math.round((liveByDest['Compost'] || 0) + (liveByDest['Composting'] || 0));
        discKg = Math.round((liveByDest['Discarded'] || 0) + (liveByDest['Discarded / Landfill'] || 0));
    } else {
        const baseWasteKg = 258.0;
        totalWasteKg = Math.round((baseWasteKg + liveWasteKg) * 10) / 10;
        const baseCookedKg = 3650.0;
        totalCookedKg = Math.round((baseCookedKg + liveWasteKg) * 10) / 10;
        totalEatenKg = Math.round((totalCookedKg - totalWasteKg) * 10) / 10;
        ngoKg = Math.round(116.0 + (liveByDest['NGO'] || 0) + (liveByDest['NGO Donation'] || 0));
        compKg = Math.round(90.0 + (liveByDest['Compost'] || 0) + (liveByDest['Composting'] || 0));
        discKg = Math.round(52.0 + (liveByDest['Discarded'] || 0) + (liveByDest['Discarded / Landfill'] || 0));
    }

    const reductionVsBaseline = Math.round(((baselineWeekKg - totalWasteKg) / baselineWeekKg) * 100);
    const savedKg = Math.max(0, baselineWeekKg - totalWasteKg);
    const moneySavedWeek = Math.round(savedKg * costPerKg);
    const costOfWaste = Math.round(totalWasteKg * costPerKg);

    return res.json({
        date: targetDate,
        data_source: dataSource,
        cost_per_kg: costPerKg,
        insufficient_data: insufficientData,
        insufficient_message: insufficientData ? "Insufficient real operational data. Submit actual meal records to generate this metric." : null,
        headcounts: headcountData,
        kpis: {
            overcookingAvoidedTodayKg: Math.round(totalAvoidedPlates * 0.35),
            wasteThisWeekKg: totalWasteKg,
            baselineWeekKg: 460,
            reductionVsBaselinePct: reductionVsBaseline,
            moneySavedWeekInr: moneySavedWeek,
            costOfWasteInr: costOfWaste,
            totalCookedKg,
            totalEatenKg,
            destinations: {
                ngoKg,
                compKg,
                discKg
            }
        }
    });
});

// Admin Planner Table Calculation
app.get('/api/admin/planner', (req, res) => {
    const day = req.query.day || 'Monday';
    const meal = req.query.meal || 'Lunch';
    const week = req.query.week || 'Week A';
    const targetDate = req.query.date || new Date().toISOString().split('T')[0];

    const hc = calculateHeadcount(targetDate, meal);
    const predHeadcount = hc.predicted_headcount;
    const fewerPlates = hc.fewer_plates;

    if (!weeklyMenuData) loadMenu();
    const dayMenu = weeklyMenuData.menu[day] || weeklyMenuData.menu['Monday'];
    const mealMenu = dayMenu[meal] || dayMenu['Lunch'];
    const rawDishes = mealMenu.dishes || ['Curd Rice', 'Dal Methi', 'Brinjal Bhartha', 'Chapati / Fulka', 'Buttermilk'];

    const items = rawDishes.map(rawDish => {
        const resolvedName = resolveDishName(rawDish, week);
        const spec = getDishPortionSpecs(resolvedName);

        let cookDisplay = '';
        let avoidDisplay = '';

        if (spec.unit === 'g') {
            const cookKg = (predHeadcount * (spec.perHeadVal / 1000)).toFixed(1);
            const avoidKg = (fewerPlates * (spec.perHeadVal / 1000)).toFixed(1);
            cookDisplay = `${cookKg} kg`;
            avoidDisplay = `${avoidKg} kg`;
        } else if (spec.unit === 'ml') {
            const cookL = (predHeadcount * (spec.perHeadVal / 1000)).toFixed(1);
            const avoidL = (fewerPlates * (spec.perHeadVal / 1000)).toFixed(1);
            cookDisplay = `${cookL} L`;
            avoidDisplay = `${avoidL} L`;
        } else {
            // Pieces (e.g. chapati)
            const cookPcs = Math.round(predHeadcount * spec.perHeadVal);
            const avoidPcs = Math.round(fewerPlates * spec.perHeadVal);
            cookDisplay = `${cookPcs} pcs`;
            avoidDisplay = `${avoidPcs} pcs`;
        }

        return {
            raw_dish: rawDish,
            dish_name: resolvedName,
            type: spec.category,
            per_head: spec.display,
            cook: cookDisplay,
            avoided: avoidDisplay
        };
    });

    return res.json({
        day,
        meal,
        week,
        predicted_headcount: predHeadcount,
        fewer_plates: fewerPlates,
        items
    });
});

// Admin Log Waste Form
app.post('/api/admin/waste', (req, res) => {
    const { meal_date, meal_type, dish_name, waste_type, weight_kg, destination, notes } = req.body;
    const wt = parseFloat(weight_kg);

    if (!meal_date || !meal_type || !dish_name || !waste_type || isNaN(wt) || wt <= 0) {
        return res.status(400).json({ success: false, message: 'Please provide valid waste details and a weight > 0 kg.' });
    }

    const nowStr = new Date().toISOString();
    const dest = destination || 'Compost';

    const stmt = db.prepare(`
        INSERT INTO waste_logs (
            meal_date, meal_type, dish_name, waste_type, weight_kg, destination, notes, recorded_by, data_source, logged_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Kitchen Staff', 'REAL', ?)
    `);
    stmt.run(meal_date, meal_type, dish_name, waste_type, wt, dest, notes || '', nowStr);

    // Auto-create alert if waste > 25 kg
    let alertCreated = false;
    if (wt > 25.0) {
        const altStmt = db.prepare(`
            INSERT INTO system_alerts (
                created_at, alert_type, title, message, severity, target_role, dish_name, meal_type, data_source, is_read, metadata_json
            ) VALUES (?, 'high_waste', ?, ?, 'warning', 'All', ?, ?, 'REAL', 0, ?)
        `);
        const title = `⚠️ High Food Waste Alert: ${dish_name}`;
        const msg = `${dish_name} logged ${wt.toFixed(1)} kg of ${waste_type} waste during ${meal_type} (destination: ${dest}). Exceeds 25 kg threshold.`;
        altStmt.run(nowStr, title, msg, dish_name, meal_type, JSON.stringify({ weight_kg: wt, destination: dest }));
        alertCreated = true;
    }

    return res.json({
        success: true,
        message: `Logged ${wt.toFixed(1)} kg ${waste_type} for ${dish_name} (${dest}).`,
        alertCreated
    });
});

// Admin Analytics (Dynamic SQLite Queries & Data Source Separation)
app.get('/api/admin/analytics', (req, res) => {
    const costPerKg = Number(req.query.cost_per_kg) || 35.0;
    const dataSource = (req.query.data_source || 'ALL').toUpperCase();

    // 1. Least popular dishes (from live dish_likes)
    const baseDishes = [
        { dish_name: 'Black Chana Sprouts', liked_pct: 38, source: 'Baseline' },
        { dish_name: 'Black Channa Masala', liked_pct: 49, source: 'Baseline' },
        { dish_name: 'Channa Dal', liked_pct: 51, source: 'Baseline' },
        { dish_name: 'Boiled Sweet Corn', liked_pct: 53, source: 'Baseline' },
        { dish_name: 'Boiled Groundnut', liked_pct: 53, source: 'Baseline' },
        { dish_name: 'Jalebi', liked_pct: 53, source: 'Baseline' }
    ];

    const likeStmt = db.prepare(`
        SELECT dish_name,
               SUM(CASE WHEN liked = 1 THEN 1 ELSE 0 END) as likes,
               COUNT(*) as total
        FROM dish_likes
        GROUP BY dish_name
        HAVING total >= 1
    `);
    const liveRows = likeStmt.all();
    let leastPopular = [];

    if (dataSource === 'REAL') {
        leastPopular = liveRows.map(r => ({
            dish_name: r.dish_name,
            liked_pct: Math.round((r.likes / r.total) * 100),
            source: 'REAL',
            total_votes: r.total
        })).sort((a, b) => a.liked_pct - b.liked_pct).slice(0, 6);
    } else {
        const liveMap = {};
        for (const r of liveRows) {
            const pct = Math.round((r.likes / r.total) * 100);
            liveMap[r.dish_name] = { dish_name: r.dish_name, liked_pct: pct, source: 'REAL', total_votes: r.total };
        }
        const merged = [];
        for (const b of baseDishes) {
            if (liveMap[b.dish_name]) {
                merged.push(liveMap[b.dish_name]);
                delete liveMap[b.dish_name];
            } else {
                merged.push(b);
            }
        }
        for (const extra of Object.values(liveMap)) {
            merged.push(extra);
        }
        merged.sort((a, b) => a.liked_pct - b.liked_pct);
        leastPopular = merged.slice(0, 6);
    }

    // 2. Most wasted dishes from waste_logs
    let mwSql = `
        SELECT dish_name, SUM(weight_kg) as total_kg, COUNT(*) as entries
        FROM waste_logs
    `;
    if (dataSource === 'REAL') {
        mwSql += " WHERE data_source = 'REAL'";
    } else if (dataSource === 'SYNTHETIC') {
        mwSql += " WHERE data_source = 'SYNTHETIC'";
    }
    mwSql += ` GROUP BY dish_name ORDER BY total_kg DESC LIMIT 6`;

    const mwStmt = db.prepare(mwSql);
    const mostWasted = mwStmt.all().map(r => ({
        dish_name: r.dish_name,
        total_kg: (Math.round(r.total_kg * 10) / 10).toFixed(1),
        entries: r.entries
    }));

    // 3. Weekly waste stats
    let wasteSql = 'SELECT COALESCE(SUM(weight_kg), 0.0) as live_waste, COUNT(*) as cnt FROM waste_logs';
    if (dataSource === 'REAL') {
        wasteSql += " WHERE data_source = 'REAL'";
    } else if (dataSource === 'SYNTHETIC') {
        wasteSql += " WHERE data_source = 'SYNTHETIC'";
    }
    const wasteRow = db.prepare(wasteSql).get();
    const liveWasteKg = Number(wasteRow?.live_waste || 0);
    const wasteCount = Number(wasteRow?.cnt || 0);

    const week4Waste = dataSource === 'REAL' ? Math.round(liveWasteKg * 10) / 10 : Math.round(258 + liveWasteKg);
    const costOfWasteInr = Math.round(week4Waste * costPerKg);
    const baselineCostInr = 16100;

    const insufficientData = (dataSource === 'REAL' && wasteCount === 0 && leastPopular.length === 0);

    return res.json({
        cost_per_kg: costPerKg,
        data_source: dataSource,
        insufficient_data: insufficientData,
        insufficient_message: insufficientData ? "Insufficient real operational data. Submit actual meal records to generate this metric." : null,
        weekly_waste: {
            week1: 460, // Baseline with no system
            week2: dataSource === 'REAL' ? 0 : 360,
            week3: dataSource === 'REAL' ? 0 : 295,
            week4: week4Waste,
            live_waste_kg: liveWasteKg
        },
        least_popular: leastPopular,
        most_wasted: mostWasted,
        cost_of_waste: {
            cost_inr: costOfWasteInr,
            baseline_inr: baselineCostInr,
            cost_per_kg: costPerKg
        }
    });
});

// Admin Broadcast Announcement
app.post('/api/admin/announcement', (req, res) => {
    const { title, message, severity, target_criteria, target_date, target_meal, sender } = req.body;
    if (!title || !message) {
        return res.status(400).json({ success: false, message: 'Title and message are required.' });
    }

    const nowStr = new Date().toISOString();
    const meta = {
        target_criteria: target_criteria || 'all',
        target_date: target_date || null,
        target_meal: target_meal || null,
        sender: sender || 'admin'
    };

    const stmt = db.prepare(`
        INSERT INTO system_alerts (
            created_at, alert_type, title, message, severity, target_role, data_source, is_read, metadata_json
        ) VALUES (?, 'announcement', ?, ?, ?, 'Student', 'REAL', 0, ?)
    `);
    stmt.run(nowStr, title.trim(), message.trim(), severity || 'info', JSON.stringify(meta));

    return res.json({
        success: true,
        message: `Announcement '${title.trim()}' broadcast successfully.`
    });
});

// Admin Reports CSV Export
app.get('/api/admin/reports/csv', (req, res) => {
    const stmt = db.prepare('SELECT * FROM waste_logs ORDER BY logged_timestamp DESC LIMIT 100');
    const rows = stmt.all();

    let csv = 'LogID,MealDate,MealType,DishName,WasteType,WeightKg,Destination,RecordedBy,Timestamp\n';
    for (const r of rows) {
        csv += `${r.log_id},"${r.meal_date}","${r.meal_type}","${r.dish_name}","${r.waste_type}",${r.weight_kg},"${r.destination}","${r.recorded_by}","${r.logged_timestamp}"\n`;
    }

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="smart_mess_audit_${new Date().toISOString().split('T')[0]}.csv"`);
    return res.send(csv);
});


// Student Eco-Impact Leaderboard API (for both Student and Admin portals)
app.get('/api/leaderboard', (req, res) => {
    try {
        const search = req.query.search ? String(req.query.search).trim() : null;
        const badgeFilter = req.query.badge || null;
        const currentRoll = req.query.roll ? String(req.query.roll).trim().toUpperCase() : null;
        const limit = Math.min(Math.max(parseInt(req.query.limit, 10) || 50, 1), 200);
        const offset = Math.max(parseInt(req.query.offset, 10) || 0, 0);

        const baseCte = `
            WITH reg_stats AS (
                SELECT 
                    UPPER(registration_code) as roll,
                    COUNT(*) as total_meals,
                    SUM(CASE WHEN attendance_intent = 'No' THEN 1 ELSE 0 END) as skips
                FROM student_registrations
                WHERE registration_code IS NOT NULL
                GROUP BY UPPER(registration_code)
            ),
            like_stats AS (
                SELECT 
                    UPPER(registration_code) as roll,
                    COUNT(*) as ratings
                FROM dish_likes
                WHERE registration_code IS NOT NULL
                GROUP BY UPPER(registration_code)
            ),
            ranked_students AS (
                SELECT 
                    u.user_id,
                    u.student_roll_number,
                    u.username,
                    COALESCE(u.full_name, u.username) as full_name,
                    COALESCE(r.total_meals, 0) as total_meals,
                    COALESCE(r.skips, 0) as meals_skipped,
                    COALESCE(l.ratings, 0) as ratings_given,
                    ROUND(COALESCE(r.skips, 0) * 0.35, 1) as food_avoided_kg,
                    (COALESCE(r.total_meals, 0) * 10 + COALESCE(l.ratings, 0) * 5 + COALESCE(r.skips, 0) * 15) as green_points,
                    ROW_NUMBER() OVER (
                        ORDER BY (COALESCE(r.total_meals, 0) * 10 + COALESCE(l.ratings, 0) * 5 + COALESCE(r.skips, 0) * 15) DESC,
                                 u.student_roll_number ASC
                    ) as rank
                FROM users u
                LEFT JOIN reg_stats r ON UPPER(u.student_roll_number) = r.roll
                LEFT JOIN like_stats l ON UPPER(u.student_roll_number) = l.roll
                WHERE u.role = 'Student'
            )
        `;

        function determineBadge(points) {
            if (points >= 500) return '🌿 Eco Champion';
            if (points >= 300) return '🌱 Eco Warrior';
            if (points >= 150) return '☘️ Sprout';
            return '🌾 Seedling';
        }

        // Summary
        const sumRow = db.prepare(baseCte + `
            SELECT 
                COUNT(*) as total_students,
                ROUND(SUM(food_avoided_kg), 1) as total_food_avoided_kg,
                SUM(green_points) as total_points
            FROM ranked_students;
        `).get();

        // Top 3
        const top3Rows = db.prepare(baseCte + ' SELECT * FROM ranked_students ORDER BY rank ASC LIMIT 3;').all().map(r => ({
            ...r,
            badge_level: determineBadge(r.green_points)
        }));

        // Current student standing
        let myStanding = null;
        if (currentRoll) {
            const myRow = db.prepare(baseCte + `
                SELECT * FROM ranked_students
                WHERE UPPER(student_roll_number) = ? OR UPPER(username) = ?
                LIMIT 1;
            `).get(currentRoll, currentRoll);
            if (myRow) {
                myStanding = {
                    ...myRow,
                    badge_level: determineBadge(myRow.green_points)
                };
            }
        }

        // Filtered rows
        const whereClauses = [];
        const params = [];
        if (search) {
            const s = `%${search.toUpperCase()}%`;
            whereClauses.push('(UPPER(student_roll_number) LIKE ? OR UPPER(username) LIKE ? OR UPPER(full_name) LIKE ?)');
            params.push(s, s, s);
        }

        const whereStr = whereClauses.length > 0 ? (' WHERE ' + whereClauses.join(' AND ')) : '';
        const countRow = db.prepare(baseCte + ` SELECT COUNT(*) as cnt FROM ranked_students ${whereStr};`).get(...params);
        const totalMatching = Number(countRow?.cnt || 0);

        const rows = db.prepare(baseCte + ` SELECT * FROM ranked_students ${whereStr} ORDER BY rank ASC LIMIT ? OFFSET ?;`).all(...params, limit, offset);

        const leaderboard = rows.map(r => ({
            ...r,
            badge_level: determineBadge(r.green_points)
        })).filter(r => {
            if (!badgeFilter || badgeFilter === 'All') return true;
            return r.badge_level.includes(badgeFilter);
        });

        return res.json({
            success: true,
            summary: {
                total_students: Number(sumRow?.total_students || 0),
                total_food_avoided_kg: Number(sumRow?.total_food_avoided_kg || 0),
                total_points: Number(sumRow?.total_points || 0),
                top_champion: top3Rows[0] || null
            },
            top_three: top3Rows,
            my_rank: myStanding,
            total: totalMatching,
            leaderboard
        });
    } catch (err) {
        console.error('Error fetching leaderboard:', err);
        return res.status(500).json({ success: false, message: 'Failed to compute leaderboard.' });
    }
});

// Export Leaderboard CSV
app.get('/api/admin/leaderboard/csv', (req, res) => {
    try {
        const baseCte = `
            WITH reg_stats AS (
                SELECT 
                    UPPER(registration_code) as roll,
                    COUNT(*) as total_meals,
                    SUM(CASE WHEN attendance_intent = 'No' THEN 1 ELSE 0 END) as skips
                FROM student_registrations
                WHERE registration_code IS NOT NULL
                GROUP BY UPPER(registration_code)
            ),
            like_stats AS (
                SELECT 
                    UPPER(registration_code) as roll,
                    COUNT(*) as ratings
                FROM dish_likes
                WHERE registration_code IS NOT NULL
                GROUP BY UPPER(registration_code)
            ),
            ranked_students AS (
                SELECT 
                    u.student_roll_number,
                    u.username,
                    COALESCE(u.full_name, u.username) as full_name,
                    COALESCE(r.total_meals, 0) as total_meals,
                    COALESCE(r.skips, 0) as meals_skipped,
                    COALESCE(l.ratings, 0) as ratings_given,
                    ROUND(COALESCE(r.skips, 0) * 0.35, 1) as food_avoided_kg,
                    (COALESCE(r.total_meals, 0) * 10 + COALESCE(l.ratings, 0) * 5 + COALESCE(r.skips, 0) * 15) as green_points,
                    ROW_NUMBER() OVER (
                        ORDER BY (COALESCE(r.total_meals, 0) * 10 + COALESCE(l.ratings, 0) * 5 + COALESCE(r.skips, 0) * 15) DESC,
                                 u.student_roll_number ASC
                    ) as rank
                FROM users u
                LEFT JOIN reg_stats r ON UPPER(u.student_roll_number) = r.roll
                LEFT JOIN like_stats l ON UPPER(u.student_roll_number) = l.roll
                WHERE u.role = 'Student'
            )
            SELECT * FROM ranked_students ORDER BY rank ASC;
        `;

        function determineBadge(pts) {
            if (pts >= 500) return 'Eco Champion';
            if (pts >= 300) return 'Eco Warrior';
            if (pts >= 150) return 'Sprout';
            return 'Seedling';
        }

        const rows = db.prepare(baseCte).all();
        let csv = 'Rank,RollNumber,Username,FullName,MealsLogged,MealsSkipped,FoodAvoidedKg,RatingsGiven,GreenPoints,EcoBadge\n';
        for (const r of rows) {
            const badge = determineBadge(r.green_points);
            const safeName = (r.full_name || '').replace(/"/g, '""');
            csv += `${r.rank},"${r.student_roll_number || ''}","${r.username}","${safeName}",${r.total_meals},${r.meals_skipped},${r.food_avoided_kg},${r.ratings_given},${r.green_points},"${badge}"\n`;
        }

        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', `attachment; filename="smart_mess_leaderboard_${new Date().toISOString().split('T')[0]}.csv"`);
        return res.send(csv);
    } catch (e) {
        return res.status(500).send('Failed to generate leaderboard CSV export.');
    }
});


// Start Express Server
if (!process.env.VERCEL) {
    app.listen(PORT, () => {
        console.log(`=======================================================`);
        console.log(`MessMind AI - Smart Mess Server running on Node.js!`);
        console.log(`Local Access: http://localhost:${PORT}`);
        console.log(`Capacity: 1,200 Students | SQLite: ${DB_PATH}`);
        console.log(`=======================================================`);
    });
}

module.exports = app;

