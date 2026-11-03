/**
 * Smart Mess - High Performance Client Application
 * Connects to Node.js Express REST APIs with live SQLite reactivity.
 */

// Global State
const state = {
    user: null,
    portal: 'Student', // 'Student' | 'Admin'
    week: 'Week A',    // 'Week A' | 'Week B'
    selectedDay: 'Mon',
    adminDay: 'Mon',
    plannerDay: 'Mon',
    plannerMeal: 'Lunch',
    rateMeal: 'Breakfast',
    costPerKg: 35,
    dataSource: 'REAL', // 'REAL' | 'ALL' | 'SYNTHETIC'
    menuData: null,
    dayMap: {
        'Mon': 'Monday', 'Tue': 'Tuesday', 'Wed': 'Wednesday',
        'Thu': 'Thursday', 'Fri': 'Friday', 'Sat': 'Saturday', 'Sun': 'Sunday'
    }
};

// Helper: Calculate exact YYYY-MM-DD for a day of current week
function getDateForDay(dayShort) {
    const dayIndices = { 'Sun': 0, 'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6 };
    const targetIdx = dayIndices[dayShort] !== undefined ? dayIndices[dayShort] : 1;
    const now = new Date();
    const currentDayIdx = now.getDay();
    const diff = targetIdx - currentDayIdx;
    const d = new Date(now);
    d.setDate(now.getDate() + diff);
    return d.toISOString().split('T')[0];
}

// ==========================================================================
// INITIALIZATION & LIFECYCLE
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    // Check saved user in localStorage
    const savedUser = localStorage.getItem('smart_mess_user');
    if (savedUser) {
        try {
            state.user = JSON.parse(savedUser);
        } catch (e) {
            state.user = null;
        }
    }

    const authModal = document.getElementById('auth-modal');
    if (!state.user) {
        if (authModal) authModal.classList.remove('hidden');
    } else {
        if (authModal) authModal.classList.add('hidden');
    }

    // Set today's short day
    const daysArr = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const todayShort = daysArr[new Date().getDay()] || 'Mon';
    const initialDay = todayShort === 'Sun' ? 'Sun' : todayShort;
    state.selectedDay = initialDay;
    state.adminDay = initialDay;
    state.plannerDay = initialDay;

    updateUserDisplay();
    initActiveWeek().finally(() => {
        fetchMenuData().then(() => {
            refreshCurrentView();
        });
    });
});

// Update Header User Profile
function updateUserDisplay() {
    const displayEl = document.getElementById('header-user-display');
    if (displayEl && state.user) {
        const rollOrName = state.user.student_roll_number || state.user.username;
        displayEl.textContent = `${rollOrName} (${state.user.role})`;
    }
    applyPortalRoleIsolation();
}

// Show Toast
function showToast(message, duration = 3000) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, duration);
}

// ==========================================================================
// AUTHENTICATION
// ==========================================================================
let currentAuthTabRole = 'Student';

function setAuthRole(role) {
    currentAuthTabRole = role;
    document.getElementById('tab-btn-student').classList.toggle('active', role === 'Student');
    document.getElementById('tab-btn-admin').classList.toggle('active', role === 'Admin');
    document.getElementById('lbl-username').textContent = role === 'Student' ? 'Student Roll Number / Username' : 'Admin Username';
    document.getElementById('auth-username').placeholder = role === 'Student' ? 'e.g. 24BDS021' : 'e.g. admin';
}

function quickFill(u, p, role) {
    setAuthRole(role);
    document.getElementById('auth-username').value = u;
    document.getElementById('auth-password').value = p;
    document.getElementById('auth-error').style.display = 'none';
}

async function handleLogin(e) {
    e.preventDefault();
    const u = document.getElementById('auth-username').value.trim();
    const p = document.getElementById('auth-password').value.trim();
    const errBox = document.getElementById('auth-error');

    try {
        const resp = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: u, password: p })
        });
        const data = await resp.json();

        if (data.success) {
            state.user = data.user;
            localStorage.setItem('smart_mess_user', JSON.stringify(data.user));
            document.getElementById('auth-modal').classList.add('hidden');
            updateUserDisplay();

            // Set portal based on role
            if (data.user.role === 'Admin') {
                switchPortal('Admin');
            } else {
                switchPortal('Student');
            }
            showToast(`Signed in as ${data.user.username}`);
        } else {
            errBox.textContent = data.message || 'Login failed.';
            errBox.style.display = 'block';
        }
    } catch (err) {
        errBox.textContent = 'Server connection error. Please try again.';
        errBox.style.display = 'block';
    }
}

function handleLogout() {
    state.user = null;
    localStorage.removeItem('smart_mess_user');
    document.getElementById('auth-modal').classList.remove('hidden');
    document.getElementById('auth-username').value = '';
    document.getElementById('auth-password').value = '';
    showToast('Signed out.');
}

// ==========================================================================
// PORTAL & TAB NAVIGATION
// ==========================================================================
function switchPortal(portal) {
    if (!state.user) {
        document.getElementById('auth-modal')?.classList.remove('hidden');
        return;
    }

    // Strict Role Enforcement
    if (portal === 'Admin' && state.user.role !== 'Admin') {
        showToast('Access Denied: Only administrators can access the Admin Portal.');
        return;
    }
    if (portal === 'Student' && state.user.role !== 'Student') {
        showToast('Access Denied: You are authenticated as an Administrator.');
        return;
    }

    state.portal = portal;
    applyPortalRoleIsolation();
    refreshCurrentView();
}

function applyPortalRoleIsolation() {
    const isStudent = (state.user && state.user.role === 'Student');
    const isAdmin = (state.user && state.user.role === 'Admin');

    const stuPill = document.getElementById('pill-portal-student');
    const admPill = document.getElementById('pill-portal-admin');
    const sPortal = document.getElementById('portal-student');
    const aPortal = document.getElementById('portal-admin');

    if (isStudent) {
        state.portal = 'Student';
        if (stuPill) {
            stuPill.style.display = 'inline-block';
            stuPill.classList.add('active');
            stuPill.textContent = '🎓 Student Portal';
        }
        if (admPill) {
            admPill.style.display = 'none'; // Hidden for students
        }
        if (sPortal) {
            sPortal.style.display = 'block';
            sPortal.classList.add('active');
        }
        if (aPortal) {
            aPortal.style.display = 'none';
            aPortal.classList.remove('active');
        }
    } else if (isAdmin) {
        state.portal = 'Admin';
        if (admPill) {
            admPill.style.display = 'inline-block';
            admPill.classList.add('active');
            admPill.textContent = '🛡️ Admin Portal';
        }
        if (stuPill) {
            stuPill.style.display = 'none'; // Hidden for admins
        }
        if (aPortal) {
            aPortal.style.display = 'block';
            aPortal.classList.add('active');
        }
        if (sPortal) {
            sPortal.style.display = 'none';
            sPortal.classList.remove('active');
        }
    }
}

async function initActiveWeek() {
    try {
        const res = await fetch('/api/config/active-week');
        const data = await res.json();
        if (data && data.active_week) {
            state.week = data.active_week;
            document.getElementById('pill-week-a')?.classList.toggle('active', data.active_week === 'Week A');
            document.getElementById('pill-week-b')?.classList.toggle('active', data.active_week === 'Week B');
            updateActiveWeekBadge(data.active_week);
        }
    } catch (e) {
        console.warn('Could not load active week:', e);
    }
}

function updateActiveWeekBadge(week) {
    const badge = document.getElementById('active-week-badge');
    if (badge) {
        badge.textContent = `Official Active: ${week}`;
    }
}

async function switchWeek(week) {
    state.week = week;
    document.getElementById('pill-week-a')?.classList.toggle('active', week === 'Week A');
    document.getElementById('pill-week-b')?.classList.toggle('active', week === 'Week B');

    if (state.user && state.user.role === 'Admin') {
        try {
            const resp = await fetch('/api/admin/active-week', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ week })
            });
            const data = await resp.json();
            if (data && data.success) {
                updateActiveWeekBadge(week);
                showToast(`Official active schedule set to ${week}`);
            }
        } catch (e) {
            console.error('Failed to update active week:', e);
        }
    } else {
        showToast(`Previewing menu for ${week}`);
    }

    refreshCurrentView();
}

function switchStudentNav(tabKey) {
    const tabs = ['weekly-menu', 'my-meals', 'rate-dishes', 'my-impact', 'leaderboard', 'notifications'];
    for (const t of tabs) {
        const btn = document.getElementById(`snav-${t}`);
        const panel = document.getElementById(`stab-${t}`);
        const isMatch = (t === tabKey);
        if (btn) btn.classList.toggle('active', isMatch);
        if (panel) {
            panel.classList.toggle('active', isMatch);
            panel.style.display = isMatch ? 'block' : 'none';
        }
    }

    if (tabKey === 'weekly-menu') renderWeeklyMenu();
    else if (tabKey === 'my-meals') renderMyMeals();
    else if (tabKey === 'rate-dishes') renderRateDishes();
    else if (tabKey === 'my-impact') renderMyImpact();
    else if (tabKey === 'leaderboard') renderStudentLeaderboard();
    else if (tabKey === 'notifications') renderStudentNotifications();
}

function switchAdminNav(tabKey) {
    const tabs = ['dashboard', 'planner', 'log-waste', 'analytics', 'leaderboard', 'users'];
    for (const t of tabs) {
        const btn = document.getElementById(`anav-${t}`);
        const panel = document.getElementById(`atab-${t}`);
        const isMatch = (t === tabKey);
        if (btn) btn.classList.toggle('active', isMatch);
        if (panel) {
            panel.classList.toggle('active', isMatch);
            panel.style.display = isMatch ? 'block' : 'none';
        }
    }

    if (tabKey === 'dashboard') renderAdminDashboard();
    else if (tabKey === 'planner') renderAdminPlanner();
    else if (tabKey === 'log-waste') updateWasteDishesList();
    else if (tabKey === 'analytics') renderAdminAnalytics();
    else if (tabKey === 'leaderboard') renderAdminLeaderboard();
    else if (tabKey === 'users') loadAdminUsers();
}

function refreshCurrentView() {
    if (state.portal === 'Student') {
        const activeTab = document.querySelector('.student-tab.active');
        if (activeTab) {
            const tabId = activeTab.id.replace('stab-', '');
            switchStudentNav(tabId);
        } else {
            switchStudentNav('weekly-menu');
        }
    } else {
        const activeTab = document.querySelector('.admin-tab.active');
        if (activeTab) {
            const tabId = activeTab.id.replace('atab-', '');
            switchAdminNav(tabId);
        } else {
            switchAdminNav('dashboard');
        }
    }
}

// ==========================================================================
// MENU DATA FETCHING & DISH RESOLUTION
// ==========================================================================
async function fetchMenuData() {
    try {
        const res = await fetch(`/api/menu?week=${encodeURIComponent(state.week)}`);
        state.menuData = await res.json();
    } catch (err) {
        console.error('Failed to fetch menu:', err);
    }
}

function resolveDish(dishStr) {
    if (!dishStr) return '';
    if (dishStr.includes(' / ')) {
        const parts = dishStr.split(' / ').map(p => p.trim());
        return state.week === 'Week B' ? (parts[1] || parts[0]) : parts[0];
    }
    if (dishStr.includes('/')) {
        const parts = dishStr.split('/').map(p => p.trim());
        return state.week === 'Week B' ? (parts[1] || parts[0]) : parts[0];
    }
    return dishStr.trim();
}

// ==========================================================================
// STUDENT TAB 1: WEEKLY MENU
// ==========================================================================
function setDay(dayShort) {
    state.selectedDay = dayShort;
    // Update day pill active state in student views
    document.querySelectorAll('#menu-day-pills .day-pill, #meals-day-pills .day-pill, #rate-day-pills .day-pill')
        .forEach(btn => {
            btn.classList.toggle('active', btn.textContent.trim() === dayShort);
        });

    const activeTab = document.querySelector('.student-tab.active');
    if (activeTab && activeTab.id === 'stab-my-meals') {
        renderMyMeals();
    } else if (activeTab && activeTab.id === 'stab-rate-dishes') {
        renderRateDishes();
    } else {
        renderWeeklyMenu();
    }
}

function renderWeeklyMenu() {
    if (!state.menuData?.menu) return;
    const fullDay = state.dayMap[state.selectedDay] || 'Monday';
    const dayMenu = state.menuData.menu[fullDay] || state.menuData.menu['Monday'];
    const grid = document.getElementById('menu-meal-grid');
    if (!grid) return;

    const meals = ['Breakfast', 'Lunch', 'Snacks', 'Dinner'];
    grid.innerHTML = meals.map(mealKey => {
        const mealObj = dayMenu[mealKey] || {};
        const dishes = mealObj.dishes || [];
        const fixed = mealObj.daily_fixed || '';

        const dishItems = dishes.map(d => `<li>${resolveDish(d)}</li>`).join('');

        return `
            <div class="meal-card">
                <div>
                    <h3 class="meal-title">${mealKey}</h3>
                    <ul class="meal-dishes-list">
                        ${dishItems}
                    </ul>
                </div>
                ${fixed ? `<div class="daily-fixed-note">Every day: ${fixed}</div>` : ''}
            </div>
        `;
    }).join('');
}

// ==========================================================================
// STUDENT TAB 2: MY MEALS (Eating / Skipping)
// ==========================================================================
async function renderMyMeals() {
    if (!state.menuData?.menu) return;
    const fullDay = state.dayMap[state.selectedDay] || 'Monday';
    const dayMenu = state.menuData.menu[fullDay] || state.menuData.menu['Monday'];
    const grid = document.getElementById('my-meals-grid');
    if (!grid) return;

    // Fetch student's recorded attendance from backend
    let mealStatus = {};
    try {
        const res = await fetch(`/api/student/dashboard?roll=${state.user.username}`);
        const data = await res.json();
        mealStatus = data.mealStatus || {};

        // Update impact card numbers in state
        updateImpactDisplay(data.impact);
        updateUnreadBadge(data.unreadNotifications);
    } catch (e) {
        console.error('Error fetching student status:', e);
    }

    const meals = ['Breakfast', 'Lunch', 'Snacks', 'Dinner'];
    grid.innerHTML = meals.map(mealKey => {
        const mealObj = dayMenu[mealKey] || {};
        const dishes = (mealObj.dishes || []).map(d => resolveDish(d)).join(', ');

        const currentIntent = mealStatus[mealKey]; // 'Yes' or 'No' or undefined
        const isEating = currentIntent === 'Yes';
        const isSkipping = currentIntent === 'No';

        return `
            <div class="meal-card">
                <div>
                    <h3 class="meal-title">${mealKey}</h3>
                    <div style="font-size:0.88rem; color:#8FA498; line-height:1.45; margin-bottom:16px;">
                        ${dishes}
                    </div>
                </div>
                <div class="eating-skipping-row">
                    <button type="button" class="btn-choice eating ${isEating ? 'selected' : ''}" 
                        onclick="toggleAttendance('${mealKey}', 'Yes')">
                        Eating
                    </button>
                    <button type="button" class="btn-choice skipping ${isSkipping ? 'selected' : ''}" 
                        onclick="toggleAttendance('${mealKey}', 'No')">
                        Skipping
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

async function toggleAttendance(meal, intent) {
    const todayStr = new Date().toISOString().split('T')[0];
    try {
        const resp = await fetch('/api/student/attendance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                roll: state.user.username,
                date: todayStr,
                meal,
                intent
            })
        });
        const data = await resp.json();
        if (data.success) {
            showToast(`${meal}: Marked as ${intent === 'Yes' ? 'Eating 👍' : 'Skipping ⏭️'}`);
            renderMyMeals();
        }
    } catch (err) {
        showToast('Failed to update attendance.');
    }
}

// ==========================================================================
// STUDENT TAB 3: RATE DISHES
// ==========================================================================
function setRateMeal(meal) {
    state.rateMeal = meal;
    document.querySelectorAll('.meal-filter-bar .sub-pill').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.trim() === meal);
    });
    renderRateDishes();
}

function renderRateDishes() {
    if (!state.menuData?.menu) return;
    const fullDay = state.dayMap[state.selectedDay] || 'Monday';
    const dayMenu = state.menuData.menu[fullDay] || state.menuData.menu['Monday'];
    const mealObj = dayMenu[state.rateMeal] || {};
    const rawDishes = mealObj.dishes || [];
    const container = document.getElementById('rate-dishes-list');
    if (!container) return;

    if (rawDishes.length === 0) {
        container.innerHTML = '<div style="color:#8FA498; padding:20px 0;">No dishes found for this meal shift.</div>';
        return;
    }

    container.innerHTML = rawDishes.map(d => {
        const resolved = resolveDish(d);
        return `
            <div class="rate-dish-row">
                <span class="rate-dish-name">${resolved}</span>
                <div class="rate-thumbs-group">
                    <button type="button" class="btn-thumb" title="Thumbs Up" onclick="rateDish('${resolved}', '${state.rateMeal}', 1, this)">👍</button>
                    <button type="button" class="btn-thumb" title="Thumbs Down" onclick="rateDish('${resolved}', '${state.rateMeal}', 0, this)">👎</button>
                </div>
            </div>
        `;
    }).join('');
}

async function rateDish(dish, meal, liked, btnEl) {
    try {
        const resp = await fetch('/api/student/rating', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                roll: state.user.username,
                dish,
                meal,
                date: new Date().toISOString().split('T')[0],
                liked
            })
        });
        const data = await resp.json();
        if (data.success) {
            // Toggle active style on parent button group
            const parent = btnEl.parentElement;
            parent.querySelectorAll('.btn-thumb').forEach(b => b.classList.remove('active'));
            btnEl.classList.add('active');
            showToast(`${dish}: Rating saved (${data.liked_pct}% liked overall)`);
        }
    } catch (e) {
        showToast('Error saving dish rating.');
    }
}

// ==========================================================================
// STUDENT TAB 4: MY IMPACT
// ==========================================================================
async function renderMyImpact() {
    try {
        const res = await fetch(`/api/student/dashboard?roll=${state.user.username}`);
        const data = await res.json();
        updateImpactDisplay(data.impact);
    } catch (e) {
        console.error('Error fetching impact:', e);
    }
}

function updateImpactDisplay(impact) {
    if (!impact) return;
    const gpEl = document.getElementById('impact-green-points');
    const lvlEl = document.getElementById('impact-level');
    const msEl = document.getElementById('impact-meals-skipped');
    const fsEl = document.getElementById('impact-food-saved');

    if (gpEl) gpEl.textContent = impact.greenPoints;
    if (lvlEl) lvlEl.textContent = `Level: ${impact.level}`;
    if (msEl) msEl.textContent = impact.mealsSkipped;
    if (fsEl) fsEl.textContent = `${impact.foodAvoidedKg} kg`;
}

function updateUnreadBadge(count) {
    const badge = document.getElementById('student-unread-badge');
    if (!badge) return;
    if (count > 0) {
        badge.textContent = count;
        badge.style.display = 'inline-block';
    } else {
        badge.style.display = 'none';
    }
}

// ==========================================================================
// STUDENT TAB 5: NOTIFICATIONS
// ==========================================================================
async function renderStudentNotifications() {
    const list = document.getElementById('student-notif-list');
    if (!list) return;

    try {
        const res = await fetch(`/api/student/notifications?roll=${state.user.username}`);
        const data = await res.json();
        const notifs = data.notifications || [];
        updateUnreadBadge(data.unreadCount);

        if (notifs.length === 0) {
            list.innerHTML = `
                <div class="meal-card" style="text-align:center; padding:40px;">
                    <div style="font-size:2rem; margin-bottom:8px;">🎉</div>
                    <div style="font-family:var(--font-serif); font-size:1.25rem; color:#FFFFFF;">No announcements at this time</div>
                    <div style="color:#8FA498; font-size:0.86rem; margin-top:4px;">You are completely up to date with mess schedule changes.</div>
                </div>
            `;
            return;
        }

        list.innerHTML = notifs.map(n => {
            const isUnread = !n.is_read;
            const sevClass = n.severity === 'critical' ? 'sev-critical' : (n.severity === 'warning' ? 'sev-warning' : '');
            const timeStr = n.created_at ? n.created_at.slice(0, 16).replace('T', ' ') : '';

            return `
                <div class="notif-item ${isUnread ? 'unread' : ''} ${sevClass}">
                    <div class="notif-body">
                        <div class="notif-title-row">
                            <span class="notif-title">${n.title}</span>
                            ${isUnread ? '<span class="pill-new">NEW</span>' : '<span class="pill-read">READ</span>'}
                        </div>
                        <div class="notif-desc">${n.message}</div>
                        <div class="notif-time">${timeStr}</div>
                    </div>
                    ${isUnread ? `
                        <button type="button" class="btn-secondary" onclick="markNotificationRead(${n.id})">
                            Mark read
                        </button>
                    ` : ''}
                </div>
            `;
        }).join('');

    } catch (e) {
        list.innerHTML = '<div style="color:#EF4444; padding:20px;">Failed to load announcements.</div>';
    }
}

async function markNotificationRead(id) {
    try {
        await fetch('/api/student/notifications/read', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ roll: state.user.username, alert_id: id })
        });
        renderStudentNotifications();
    } catch (e) {
        showToast('Error updating notification.');
    }
}

async function markAllNotificationsRead() {
    try {
        await fetch('/api/student/notifications/read', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ roll: state.user.username, all: true })
        });
        showToast('All notifications marked as read.');
        renderStudentNotifications();
    } catch (e) {
        showToast('Error updating notifications.');
    }
}

// ==========================================================================
// ADMIN PORTAL: DASHBOARD
// ==========================================================================
function setAdminDay(dayShort) {
    state.adminDay = dayShort;
    document.querySelectorAll('#admin-day-pills .day-pill').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.trim() === dayShort);
    });
    const fullDay = state.dayMap[dayShort] || 'Monday';
    const heading = document.getElementById('admin-headcount-heading');
    if (heading) heading.textContent = `Expected headcount for ${fullDay}`;
    renderAdminDashboard();
}

function setDataSource(source) {
    state.dataSource = source;
    // Update pills on both dashboard and analytics
    document.querySelectorAll('.data-source-bar .ds-pill').forEach(btn => {
        const text = btn.textContent.trim();
        const isActive = (source === 'REAL' && text === 'REAL') ||
                         (source === 'ALL' && text.startsWith('ALL')) ||
                         (source === 'SYNTHETIC' && text === 'SYNTHETIC');
        btn.classList.toggle('active', isActive);
    });
    renderAdminDashboard();
    renderAdminAnalytics();
}

async function renderAdminDashboard() {
    const targetDate = getDateForDay(state.adminDay);
    try {
        const res = await fetch(`/api/admin/dashboard?date=${targetDate}&cost_per_kg=${state.costPerKg}&data_source=${state.dataSource}`);
        const data = await res.json();

        // Handle Insufficient Real Data Alert
        const noticeEl = document.getElementById('dashboard-insufficient-notice');
        if (noticeEl) {
            if (data.insufficient_data) {
                noticeEl.style.display = 'flex';
                noticeEl.innerHTML = `<span>⚠️</span><span>${data.insufficient_message || 'Insufficient real operational data. Submit actual meal records to generate this metric.'}</span>`;
            } else {
                noticeEl.style.display = 'none';
            }
        }

        // 1. Render 4 Headcount Cards
        const grid = document.getElementById('headcount-cards-grid');
        if (grid && data.headcounts) {
            const meals = ['Breakfast', 'Lunch', 'Snacks', 'Dinner'];
            grid.innerHTML = meals.map(m => {
                const hc = data.headcounts[m] || {};
                return `
                    <div class="headcount-card">
                        <div>
                            <div class="hc-meal-name">${m}</div>
                            <div class="hc-count">${hc.predicted_headcount || 0}</div>
                            <div class="hc-breakdown">
                                ${hc.replied} of 1,200 replied (${hc.eating} eating). Rest estimated at ${hc.rate_pct}% from the 7-day average.
                            </div>
                        </div>
                        <div class="hc-avoided-plates">
                            ${hc.fewer_plates} fewer plates than cooking for everyone
                        </div>
                    </div>
                `;
            }).join('');
        }

        // 2. Render 4 KPI Metrics
        const kpis = data.kpis || {};
        const ocEl = document.getElementById('kpi-overcooking-avoided');
        const wwEl = document.getElementById('kpi-waste-this-week');
        const redEl = document.getElementById('kpi-reduction-pct');
        const msEl = document.getElementById('kpi-money-saved');
        const mslEl = document.getElementById('kpi-money-saved-label');

        if (ocEl) ocEl.textContent = `${kpis.overcookingAvoidedTodayKg} kg`;
        if (wwEl) wwEl.textContent = `${kpis.wasteThisWeekKg} kg`;
        if (redEl) redEl.textContent = `${kpis.reductionVsBaselinePct}%`;
        if (msEl) msEl.textContent = `₹${kpis.moneySavedWeekInr?.toLocaleString()}`;
        if (mslEl) mslEl.textContent = `Money saved per week (₹${state.costPerKg}/kg)`;

        // 3. Render Food Flow Bars
        const summaryEl = document.getElementById('flow-cooked-summary');
        if (summaryEl) {
            summaryEl.innerHTML = `Cooked ${kpis.totalCookedKg?.toLocaleString()} kg, of which <strong style="color:var(--mint-primary);">${kpis.totalEatenKg?.toLocaleString()} kg</strong> was eaten and <strong style="color:var(--accent-red);">${kpis.wasteThisWeekKg} kg</strong> wasted.`;
        }

        const safeCooked = Math.max(1, kpis.totalCookedKg || 1);
        const eatenPct = Math.round((kpis.totalEatenKg / safeCooked) * 100) || 93;
        const wastePct = Math.max(0, 100 - eatenPct);
        const barEaten = document.getElementById('flow-bar-eaten');
        const barWaste = document.getElementById('flow-bar-waste');
        if (barEaten) barEaten.style.width = `${eatenPct}%`;
        if (barWaste) barWaste.style.width = `${wastePct}%`;

        // Destination breakdown
        const dest = kpis.destinations || { ngoKg: 116, compKg: 90, discKg: 52 };
        const totalW = Math.max(1, dest.ngoKg + dest.compKg + dest.discKg);
        const ngoPct = Math.round((dest.ngoKg / totalW) * 100);
        const compPct = Math.round((dest.compKg / totalW) * 100);
        const discPct = Math.max(0, 100 - (ngoPct + compPct));

        document.getElementById('flow-bar-ngo').style.width = `${ngoPct}%`;
        document.getElementById('flow-bar-compost').style.width = `${compPct}%`;
        document.getElementById('flow-bar-disc').style.width = `${discPct}%`;

        document.getElementById('flow-txt-ngo').textContent = `${dest.ngoKg} kg (${ngoPct}%)`;
        document.getElementById('flow-txt-comp').textContent = `${dest.compKg} kg (${compPct}%)`;
        document.getElementById('flow-txt-disc').textContent = `${dest.discKg} kg (${discPct}%)`;

        document.getElementById('flow-sub-title').textContent = `What happened to the wasted ${kpis.wasteThisWeekKg} kg`;

    } catch (e) {
        console.error('Error rendering dashboard:', e);
    }
}

// Broadcaster Controls
function toggleBroadcaster() {
    const body = document.getElementById('broadcaster-body');
    const icon = document.getElementById('broadcaster-toggle-icon');
    if (!body) return;
    const isHidden = body.style.display === 'none';
    body.style.display = isHidden ? 'block' : 'none';
    if (icon) icon.textContent = isHidden ? '▲' : '▼';
}

function applyBroadcastTemplate() {
    const tpl = document.getElementById('bc-template-select').value;
    const titleInput = document.getElementById('bc-title');
    const msgInput = document.getElementById('bc-message');

    titleInput.value = tpl;
    if (tpl === "Tomorrow's breakfast menu has changed.") {
        msgInput.value = 'Please check the updated Breakfast menu under Weekly Menu tab before the 8 PM cutoff.';
    } else if (tpl === 'Lunch starts at 12:30 PM.') {
        msgInput.value = 'Dining hall doors for Lunch will open at 12:30 PM today. Counter service runs until 2:30 PM.';
    } else if (tpl === 'Dinner service timing has been updated.') {
        msgInput.value = 'Dinner service will operate from 7:30 PM to 9:45 PM tonight.';
    } else if (tpl === 'Please submit your attendance before 8 PM.') {
        msgInput.value = "Kindly update your Eating/Skipping preferences for tomorrow's meals before 8 PM so the kitchen cooks exact quantities.";
    } else {
        msgInput.value = '';
    }
}

async function sendBroadcastAnnouncement() {
    const title = document.getElementById('bc-title').value.trim();
    const msg = document.getElementById('bc-message').value.trim();
    const sev = document.getElementById('bc-severity').value;
    const crit = document.getElementById('bc-targeting').value;
    const meal = document.getElementById('bc-meal').value;

    if (!title || !msg) {
        showToast('Please enter both title and message body.');
        return;
    }

    try {
        const resp = await fetch('/api/admin/announcement', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title,
                message: msg,
                severity: sev,
                target_criteria: crit,
                target_date: new Date().toISOString().split('T')[0],
                target_meal: meal,
                sender: state.user?.username || 'admin'
            })
        });
        const data = await resp.json();
        if (data.success) {
            showToast('Announcement broadcast successfully!');
            document.getElementById('bc-title').value = '';
            document.getElementById('bc-message').value = '';
            toggleBroadcaster();
        }
    } catch (e) {
        showToast('Error broadcasting announcement.');
    }
}

// ==========================================================================
// ADMIN PORTAL: PLANNER
// ==========================================================================
function setPlannerDay(dayShort) {
    state.plannerDay = dayShort;
    document.querySelectorAll('#planner-day-pills .day-pill').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.trim() === dayShort);
    });
    renderAdminPlanner();
}

function setPlannerMeal(meal) {
    state.plannerMeal = meal;
    document.querySelectorAll('#planner-meal-pills .sub-pill').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.trim() === meal);
    });
    renderAdminPlanner();
}

async function renderAdminPlanner() {
    const fullDay = state.dayMap[state.plannerDay] || 'Monday';
    const meal = state.plannerMeal || 'Lunch';
    const tbody = document.getElementById('planner-table-body');
    const headerTitle = document.getElementById('planner-header-title');
    if (!tbody) return;

    try {
        const res = await fetch(`/api/admin/planner?day=${fullDay}&meal=${meal}&week=${encodeURIComponent(state.week)}`);
        const data = await res.json();

        if (headerTitle) {
            headerTitle.textContent = `${fullDay} ${meal}: cook for ${data.predicted_headcount} students`;
        }

        const items = data.items || [];
        tbody.innerHTML = items.map(it => `
            <tr>
                <td style="font-weight:600;">${it.dish_name}</td>
                <td class="type-col">${it.type}</td>
                <td>${it.per_head}</td>
                <td style="font-weight:700;">${it.cook}</td>
                <td class="avoided-col">${it.avoided}</td>
            </tr>
        `).join('');

    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="5" style="color:#EF4444;">Failed to load planner data.</td></tr>';
    }
}

// ==========================================================================
// ADMIN PORTAL: LOG WASTE
// ==========================================================================
function updateWasteDishesList() {
    const mealSelect = document.getElementById('waste-meal-select');
    const dishSelect = document.getElementById('waste-dish-select');
    if (!mealSelect || !dishSelect || !state.menuData?.menu) return;

    const fullDay = state.dayMap[state.adminDay] || 'Monday';
    const dayMenu = state.menuData.menu[fullDay] || state.menuData.menu['Monday'];
    const chosenMeal = mealSelect.value;
    const rawDishes = dayMenu[chosenMeal]?.dishes || [];

    dishSelect.innerHTML = rawDishes.map(d => {
        const resolved = resolveDish(d);
        return `<option value="${resolved}">${resolved}</option>`;
    }).join('');
}

async function handleWasteSubmit(e) {
    e.preventDefault();
    const mealDate = new Date().toISOString().split('T')[0];
    const mealType = document.getElementById('waste-meal-select').value;
    const dishName = document.getElementById('waste-dish-select').value;
    const wasteType = document.getElementById('waste-type-select').value;
    const weightKg = document.getElementById('waste-weight-input').value;
    const destination = document.getElementById('waste-dest-select').value;
    const notes = document.getElementById('waste-notes-input').value;

    try {
        const resp = await fetch('/api/admin/waste', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                meal_date: mealDate,
                meal_type: mealType,
                dish_name: dishName,
                waste_type: wasteType,
                weight_kg: weightKg,
                destination,
                notes
            })
        });
        const data = await resp.json();
        if (data.success) {
            showToast(data.message);
            document.getElementById('waste-weight-input').value = '';
            document.getElementById('waste-notes-input').value = '';

            if (data.alertCreated) {
                showToast('⚠️ Alert: Entry exceeds 25 kg threshold!', 4000);
            }
        } else {
            showToast(data.message || 'Error saving waste log.');
        }
    } catch (err) {
        showToast('Server error while saving waste log.');
    }
}

// ==========================================================================
// ADMIN PORTAL: ANALYTICS (EXACT MATCH TO REFERENCE SCREENSHOT)
// ==========================================================================
async function renderAdminAnalytics() {
    try {
        const res = await fetch(`/api/admin/analytics?cost_per_kg=${state.costPerKg}&data_source=${state.dataSource}`);
        const data = await res.json();

        // Handle Insufficient Real Data Alert
        const noticeEl = document.getElementById('analytics-insufficient-notice');
        if (noticeEl) {
            if (data.insufficient_data) {
                noticeEl.style.display = 'flex';
                noticeEl.innerHTML = `<span>⚠️</span><span>${data.insufficient_message || 'Insufficient real operational data. Submit actual meal records to generate this metric.'}</span>`;
            } else {
                noticeEl.style.display = 'none';
            }
        }

        // 1. Column 1: Weekly Waste Bar Chart (Update Week 4 Label)
        const w4Val = data.weekly_waste?.week4 || 0;
        const w4Label = document.getElementById('svg-w4-label');
        if (w4Label) w4Label.textContent = `${w4Val} kg`;

        // 2. Column 2: Least Popular Dishes
        const lpList = document.getElementById('analytics-least-popular-list');
        if (lpList) {
            if (!data.least_popular || data.least_popular.length === 0) {
                lpList.innerHTML = '<div class="empty-wasted-msg">Insufficient real operational data. Submit actual dish ratings to generate this metric.</div>';
            } else {
                lpList.innerHTML = data.least_popular.map(it => `
                    <div class="dish-rank-item">
                        <span class="rank-dish-name">
                            ${it.dish_name}
                            ${it.source === 'REAL' ? '<span style="font-size:0.7rem; color:var(--mint-primary); margin-left:4px;">● live</span>' : ''}
                        </span>
                        <span class="rank-dish-pct">${it.liked_pct}% liked</span>
                    </div>
                `).join('');
            }
        }

        // 3. Column 3: Most Wasted Dishes (Your entries)
        const mwList = document.getElementById('analytics-most-wasted-list');
        const mwNote = document.getElementById('analytics-wasted-note');
        if (mwList) {
            if (!data.most_wasted || data.most_wasted.length === 0) {
                mwList.innerHTML = '<div class="empty-wasted-msg">Insufficient real operational data. Submit actual meal records to generate this metric.</div>';
                if (mwNote) mwNote.textContent = 'Updated immediately with each waste log.';
            } else {
                mwList.innerHTML = data.most_wasted.map(it => `
                    <div class="wasted-dish-item">
                        <span style="color:#FFFFFF;">${it.dish_name}</span>
                        <span class="wasted-kg">${it.total_kg} kg</span>
                    </div>
                `).join('');
                if (mwNote) mwNote.textContent = 'Aggregated live from your SQLite entries.';
            }
        }

        // 4. Column 4: Cost of Waste Card
        const costFig = document.getElementById('analytics-cost-figure');
        const costSub = document.getElementById('analytics-cost-subtext');
        const costInput = document.getElementById('analytics-cost-input');

        if (costFig) costFig.textContent = `₹${data.cost_of_waste?.cost_inr?.toLocaleString() || '0'}`;
        if (costSub) costSub.textContent = `This week at ₹${state.costPerKg}/kg. Baseline week was ₹16,100.`;
        if (costInput && document.activeElement !== costInput) costInput.value = state.costPerKg;

        // 5. Environmental Audit Section
        const ww = data.weekly_waste?.week4 || 0;
        const liveW = data.weekly_waste?.live_waste_kg || 0;
        const ngoW = Math.round(116 + (liveW * 0.45));
        const compW = Math.round(90 + (liveW * 0.35));
        const discW = Math.max(0, ww - (ngoW + compW));

        const eNgo = document.getElementById('ea-ngo-val');
        const eComp = document.getElementById('ea-comp-val');
        const eDisc = document.getElementById('ea-disc-val');

        if (eNgo) eNgo.textContent = `${ngoW} kg`;
        if (eComp) eComp.textContent = `${compW} kg`;
        if (eDisc) eDisc.textContent = `${discW} kg`;

    } catch (e) {
        console.error('Error rendering analytics:', e);
    }
}

let costDebounceTimer = null;
function updateCostPerKg(newVal) {
    const val = Number(newVal);
    if (isNaN(val) || val <= 0) return;
    state.costPerKg = val;

    clearTimeout(costDebounceTimer);
    costDebounceTimer = setTimeout(() => {
        renderAdminAnalytics();
        renderAdminDashboard();
    }, 250);
}

// ==========================================================================
// ADMIN PORTAL: USER MANAGEMENT
// ==========================================================================
const userDirectoryState = {
    search: '',
    role: 'All',
    status: 'All',
    page: 1,
    limit: 15,
    total: 0
};

let userSearchDebounceTimer = null;

function toggleRollNumberField() {
    const roleSelect = document.getElementById('new-user-role');
    const rollGroup = document.getElementById('group-new-roll');
    const rollInput = document.getElementById('new-user-roll');
    if (!roleSelect || !rollGroup) return;

    const isStudent = roleSelect.value === 'Student';
    rollGroup.style.display = isStudent ? 'block' : 'none';
    if (rollInput) {
        rollInput.required = isStudent;
    }
}

function toggleAddUserForm() {
    const card = document.getElementById('add-user-card');
    if (!card) return;
    const isHidden = card.style.display === 'none';
    card.style.display = isHidden ? 'block' : 'none';
    if (isHidden) {
        document.getElementById('new-user-name')?.focus();
    }
}

function handleUserSearchInput(val) {
    clearTimeout(userSearchDebounceTimer);
    userSearchDebounceTimer = setTimeout(() => {
        userDirectoryState.search = val.trim();
        userDirectoryState.page = 1;
        loadAdminUsers();
    }, 280);
}

function handleUserFilterChange() {
    userDirectoryState.role = document.getElementById('user-role-filter')?.value || 'All';
    userDirectoryState.status = document.getElementById('user-status-filter')?.value || 'All';
    userDirectoryState.page = 1;
    loadAdminUsers();
}

function prevUserPage() {
    if (userDirectoryState.page > 1) {
        userDirectoryState.page--;
        loadAdminUsers();
    }
}

function nextUserPage() {
    const maxPage = Math.ceil(userDirectoryState.total / userDirectoryState.limit) || 1;
    if (userDirectoryState.page < maxPage) {
        userDirectoryState.page++;
        loadAdminUsers();
    }
}

async function loadUserStats() {
    try {
        const res = await fetch('/api/admin/users/stats');
        const data = await res.json();
        if (data && data.success && data.stats) {
            const s = data.stats;
            const elStudents = document.getElementById('stat-total-students');
            const elStaff = document.getElementById('stat-total-staff');
            const elAdmins = document.getElementById('stat-total-admins');
            const elActive = document.getElementById('stat-active-accounts');

            if (elStudents) elStudents.textContent = Number(s.students || 0).toLocaleString();
            if (elStaff) elStaff.textContent = Number(s.kitchen_staff || 0).toLocaleString();
            if (elAdmins) elAdmins.textContent = Number(s.admins || 0).toLocaleString();
            if (elActive) elActive.textContent = Number(s.active || 0).toLocaleString();
        }
    } catch (e) {
        console.warn('Could not fetch user stats:', e);
    }
}

async function loadAdminUsers() {
    loadUserStats();

    const tbody = document.getElementById('users-table-body');
    const countInfo = document.getElementById('user-count-info');
    const pageIndicator = document.getElementById('user-page-indicator');
    const prevBtn = document.getElementById('btn-prev-page');
    const nextBtn = document.getElementById('btn-next-page');

    if (!tbody) return;

    const offset = (userDirectoryState.page - 1) * userDirectoryState.limit;
    const url = `/api/admin/users?search=${encodeURIComponent(userDirectoryState.search)}&role=${encodeURIComponent(userDirectoryState.role)}&status=${encodeURIComponent(userDirectoryState.status)}&limit=${userDirectoryState.limit}&offset=${offset}`;

    try {
        const res = await fetch(url);
        const data = await res.json();
        const users = data.users || [];
        const total = data.total || 0;
        userDirectoryState.total = total;

        if (users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:#8FA498; padding:24px;">No user accounts found matching your query.</td></tr>';
        } else {
            tbody.innerHTML = users.map(u => {
                const isActive = (u.account_status === 'Active');
                const safeName = (u.full_name || '').replace(/'/g, "\'");
                const safeUName = (u.username || '').replace(/'/g, "\'");
                const safeRoll = (u.student_roll_number || '').replace(/'/g, "\'");
                const safeRole = (u.role || '').replace(/'/g, "\'");

                return `
                    <tr>
                        <td style="color:var(--mint-primary); font-weight:600;">${u.student_roll_number || '—'}</td>
                        <td style="font-weight:600; color:#FFFFFF;">${u.username}</td>
                        <td style="color:#CBD5E1;">${u.full_name || '—'}</td>
                        <td><span class="pill-read" style="background:#13271E; color:#A7F3D0; border:1px solid rgba(52,211,153,0.3);">${u.role}</span></td>
                        <td><span class="${isActive ? 'badge-active' : 'badge-inactive'}">${u.account_status || 'Active'}</span></td>
                        <td>
                            <button type="button" class="tbl-action-btn" onclick="openEditUserModal(${u.user_id}, '${safeUName}', '${safeName}', '${safeRoll}', '${safeRole}')">✏️ Edit</button>
                            <button type="button" class="tbl-action-btn ${isActive ? 'btn-deactivate' : 'btn-activate'}" onclick="handleToggleUserStatus(${u.user_id}, '${u.account_status || 'Active'}')">
                                ${isActive ? 'Deactivate' : 'Activate'}
                            </button>
                            <button type="button" class="tbl-action-btn" onclick="openResetPasswordModal(${u.user_id}, '${safeUName}')">🔒 Reset Pwd</button>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        const maxPage = Math.ceil(total / userDirectoryState.limit) || 1;
        if (countInfo) {
            const startNum = total === 0 ? 0 : offset + 1;
            const endNum = Math.min(offset + users.length, total);
            countInfo.textContent = `Showing ${startNum}–${endNum} of ${total} accounts`;
        }
        if (pageIndicator) {
            pageIndicator.textContent = `Page ${userDirectoryState.page} of ${maxPage}`;
        }
        if (prevBtn) prevBtn.disabled = (userDirectoryState.page <= 1);
        if (nextBtn) nextBtn.disabled = (userDirectoryState.page >= maxPage);

    } catch (e) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:#EF4444; padding:20px; text-align:center;">Failed to load user accounts from database.</td></tr>';
    }
}

async function handleCreateUser(e) {
    e.preventDefault();
    const role = document.getElementById('new-user-role').value;
    const student_roll_number = document.getElementById('new-user-roll')?.value.trim();
    const username = document.getElementById('new-user-name').value.trim();
    const full_name = document.getElementById('new-user-fullname').value.trim();
    const password = document.getElementById('new-user-password').value.trim();

    if (!username || !password) {
        showToast('Username and password are required.');
        return;
    }
    if (role === 'Student' && !student_roll_number) {
        showToast('Student roll number is required for student accounts.');
        return;
    }

    try {
        const resp = await fetch('/api/admin/users/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ role, student_roll_number, username, full_name, password })
        });
        const data = await resp.json();

        if (data.success) {
            showToast(`User ${data.user.username} created successfully!`);
            document.getElementById('create-user-form').reset();
            toggleRollNumberField();
            toggleAddUserForm();
            loadAdminUsers();
        } else {
            showToast(data.message || 'Error creating user account.');
        }
    } catch (err) {
        showToast('Server connection error while creating user.');
    }
}

function openEditUserModal(uid, username, fullname, roll, role) {
    document.getElementById('edit-user-id').value = uid;
    document.getElementById('edit-user-username').value = username;
    document.getElementById('edit-user-fullname').value = fullname;
    document.getElementById('edit-user-roll').value = roll;
    document.getElementById('edit-user-role').value = role;
    document.getElementById('edit-user-modal').classList.remove('hidden');
}

function closeEditUserModal() {
    document.getElementById('edit-user-modal').classList.add('hidden');
}

async function handleSaveEditUser(e) {
    e.preventDefault();
    const user_id = document.getElementById('edit-user-id').value;
    const full_name = document.getElementById('edit-user-fullname').value.trim();
    const student_roll_number = document.getElementById('edit-user-roll').value.trim();
    const role = document.getElementById('edit-user-role').value;

    try {
        const resp = await fetch('/api/admin/users/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id, full_name, student_roll_number, role })
        });
        const data = await resp.json();
        if (data.success) {
            showToast('User details updated successfully');
            closeEditUserModal();
            loadAdminUsers();
        } else {
            showToast(data.message || 'Update failed');
        }
    } catch (err) {
        showToast('Server error while updating user');
    }
}

async function handleToggleUserStatus(uid, currentStatus) {
    const newStatus = (currentStatus === 'Active') ? 'Inactive' : 'Active';
    const actionLabel = (newStatus === 'Active') ? 'activate' : 'deactivate';

    if (!confirm(`Are you sure you want to ${actionLabel} this account?`)) {
        return;
    }

    try {
        const resp = await fetch('/api/admin/users/toggle-status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: uid, account_status: newStatus })
        });
        const data = await resp.json();
        if (data.success) {
            showToast(`Account status updated to ${newStatus}`);
            loadAdminUsers();
        } else {
            showToast(data.message || 'Action failed');
        }
    } catch (err) {
        showToast('Server error while updating status');
    }
}

function openResetPasswordModal(uid, username) {
    document.getElementById('reset-user-id').value = uid;
    document.getElementById('reset-password-sub').textContent = `Set a new secure password for ${username}.`;
    document.getElementById('reset-new-password').value = '';
    document.getElementById('reset-confirm-password').value = '';
    document.getElementById('reset-password-modal').classList.remove('hidden');
}

function closeResetPasswordModal() {
    document.getElementById('reset-password-modal').classList.add('hidden');
}

async function handleSaveResetPassword(e) {
    e.preventDefault();
    const user_id = document.getElementById('reset-user-id').value;
    const new_password = document.getElementById('reset-new-password').value;
    const confirm_password = document.getElementById('reset-confirm-password').value;

    if (new_password.length < 6) {
        showToast('Password must be at least 6 characters long.');
        return;
    }
    if (new_password !== confirm_password) {
        showToast('Passwords do not match.');
        return;
    }

    try {
        const resp = await fetch('/api/admin/users/reset-password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id, new_password })
        });
        const data = await resp.json();
        if (data.success) {
            showToast('Password reset successfully!');
            closeResetPasswordModal();
        } else {
            showToast(data.message || 'Password reset failed.');
        }
    } catch (err) {
        showToast('Server error while resetting password.');
    }
}



// ==========================================================================
// STUDENT & ADMIN LEADERBOARDS
// ==========================================================================
const studentLbState = {
    search: '',
    page: 1,
    limit: 15,
    total: 0
};

const adminLbState = {
    search: '',
    badge: 'All',
    page: 1,
    limit: 15,
    total: 0
};

let studentLbSearchTimer = null;
let adminLbSearchTimer = null;

function handleStudentLeaderboardSearch(val) {
    clearTimeout(studentLbSearchTimer);
    studentLbSearchTimer = setTimeout(() => {
        studentLbState.search = val.trim();
        studentLbState.page = 1;
        renderStudentLeaderboard();
    }, 280);
}

function prevStudentLbPage() {
    if (studentLbState.page > 1) {
        studentLbState.page--;
        renderStudentLeaderboard();
    }
}

function nextStudentLbPage() {
    const maxPage = Math.ceil(studentLbState.total / studentLbState.limit) || 1;
    if (studentLbState.page < maxPage) {
        studentLbState.page++;
        renderStudentLeaderboard();
    }
}

async function renderStudentLeaderboard() {
    const currentRoll = state.user?.student_roll_number || state.user?.username || '';
    const offset = (studentLbState.page - 1) * studentLbState.limit;
    const url = `/api/leaderboard?roll=${encodeURIComponent(currentRoll)}&search=${encodeURIComponent(studentLbState.search)}&limit=${studentLbState.limit}&offset=${offset}`;

    const podiumContainer = document.getElementById('student-podium-grid');
    const tbody = document.getElementById('student-lb-tbody');
    const totalLabel = document.getElementById('student-lb-total-label');
    const pageInfo = document.getElementById('student-lb-page-info');
    const pageIndicator = document.getElementById('student-lb-page-indicator');
    const prevBtn = document.getElementById('btn-prev-slb');
    const nextBtn = document.getElementById('btn-next-slb');

    // Update My Standing Banner
    const msRankNum = document.getElementById('ms-rank-num');
    const msName = document.getElementById('ms-student-name');
    const msSub = document.getElementById('ms-student-sub');
    const msPts = document.getElementById('ms-green-pts');
    const msSaved = document.getElementById('ms-food-saved');
    const msSkips = document.getElementById('ms-skips-count');
    const msBadge = document.getElementById('ms-badge-tag');

    try {
        const res = await fetch(url);
        const data = await res.json();
        const top3 = data.top_three || [];
        const leaderboard = data.leaderboard || [];
        const myRank = data.my_rank || null;
        studentLbState.total = data.total || 0;

        // Render My Standing Banner
        if (myRank) {
            if (msRankNum) msRankNum.textContent = `#${myRank.rank}`;
            if (msName) msName.textContent = myRank.full_name || state.user?.username;
            if (msSub) msSub.textContent = `Roll: ${myRank.student_roll_number || '—'} • Mess 1 Dining`;
            if (msPts) msPts.textContent = `${myRank.green_points} pts`;
            if (msSaved) msSaved.textContent = `${myRank.food_avoided_kg} kg`;
            if (msSkips) msSkips.textContent = `${myRank.meals_skipped}`;
            if (msBadge) {
                msBadge.textContent = myRank.badge_level;
                msBadge.className = getBadgeClass(myRank.badge_level);
            }
        } else if (state.user) {
            if (msName) msName.textContent = state.user.username;
            if (msSub) msSub.textContent = `Student Profile • Mess 1`;
        }

        // Render Podium (Top 3)
        if (podiumContainer && top3.length >= 3) {
            const medals = ['🥇', '🥈', '🥉'];
            const rankClasses = ['rank-1', 'rank-2', 'rank-3'];
            podiumContainer.innerHTML = top3.map((st, i) => `
                <div class="podium-card ${rankClasses[i]}">
                    <span class="podium-medal">${medals[i]}</span>
                    <div class="podium-roll">${st.student_roll_number || st.username}</div>
                    <div class="podium-name">${st.full_name || 'Enrolled Student'}</div>
                    <div class="podium-pts">${st.green_points} pts</div>
                    <div class="podium-sub">Saved ${st.food_avoided_kg} kg food • ${st.meals_skipped} skips</div>
                    <div style="margin-top:8px;">
                        <span class="${getBadgeClass(st.badge_level)}">${st.badge_level}</span>
                    </div>
                </div>
            `).join('');
        }

        if (totalLabel && data.summary) {
            totalLabel.textContent = `${Number(data.summary.total_students || 0).toLocaleString()} Students • ${data.summary.total_food_avoided_kg} kg Food Saved`;
        }

        // Render Table Body
        if (tbody) {
            if (leaderboard.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:#8FA498; padding:24px;">No students found matching your search.</td></tr>';
            } else {
                tbody.innerHTML = leaderboard.map(st => {
                    const isMe = (currentRoll && (st.student_roll_number?.toUpperCase() === currentRoll || st.username?.toUpperCase() === currentRoll));
                    const rankClass = st.rank === 1 ? 'rank-top-1' : (st.rank === 2 ? 'rank-top-2' : (st.rank === 3 ? 'rank-top-3' : ''));
                    const medalIcon = st.rank === 1 ? '🥇 ' : (st.rank === 2 ? '🥈 ' : (st.rank === 3 ? '🥉 ' : ''));

                    return `
                        <tr class="${isMe ? 'current-user-row' : ''}">
                            <td><span class="rank-num-badge ${rankClass}">${medalIcon}#${st.rank}</span></td>
                            <td style="font-weight:600; color:var(--mint-primary);">${st.student_roll_number || '—'}</td>
                            <td style="color:#FFFFFF; font-weight:500;">${st.full_name || st.username} ${isMe ? '<span style="font-size:0.75rem; color:#A7F3D0; font-weight:700;">(You)</span>' : ''}</td>
                            <td>${st.total_meals}</td>
                            <td style="color:#6EE7B7; font-weight:600;">${st.food_avoided_kg} kg</td>
                            <td>${st.ratings_given}</td>
                            <td style="font-weight:700; color:#FFFFFF;">${st.green_points}</td>
                            <td><span class="${getBadgeClass(st.badge_level)}">${st.badge_level}</span></td>
                        </tr>
                    `;
                }).join('');
            }
        }

        // Pagination
        const maxPage = Math.ceil(studentLbState.total / studentLbState.limit) || 1;
        if (pageInfo) {
            const startNum = studentLbState.total === 0 ? 0 : offset + 1;
            const endNum = Math.min(offset + leaderboard.length, studentLbState.total);
            pageInfo.textContent = `Showing ${startNum}–${endNum} of ${studentLbState.total} students`;
        }
        if (pageIndicator) {
            pageIndicator.textContent = `Page ${studentLbState.page} of ${maxPage}`;
        }
        if (prevBtn) prevBtn.disabled = (studentLbState.page <= 1);
        if (nextBtn) nextBtn.disabled = (studentLbState.page >= maxPage);

    } catch (e) {
        if (tbody) tbody.innerHTML = '<tr><td colspan="8" style="color:#EF4444; padding:20px; text-align:center;">Failed to load leaderboard data.</td></tr>';
    }
}

function handleAdminLeaderboardSearch(val) {
    clearTimeout(adminLbSearchTimer);
    adminLbSearchTimer = setTimeout(() => {
        adminLbState.search = val.trim();
        adminLbState.page = 1;
        renderAdminLeaderboard();
    }, 280);
}

function handleAdminLeaderboardFilter() {
    adminLbState.badge = document.getElementById('admin-lb-badge-filter')?.value || 'All';
    adminLbState.page = 1;
    renderAdminLeaderboard();
}

function prevAdminLbPage() {
    if (adminLbState.page > 1) {
        adminLbState.page--;
        renderAdminLeaderboard();
    }
}

function nextAdminLbPage() {
    const maxPage = Math.ceil(adminLbState.total / adminLbState.limit) || 1;
    if (adminLbState.page < maxPage) {
        adminLbState.page++;
        renderAdminLeaderboard();
    }
}

async function renderAdminLeaderboard() {
    const offset = (adminLbState.page - 1) * adminLbState.limit;
    const url = `/api/leaderboard?search=${encodeURIComponent(adminLbState.search)}&badge=${encodeURIComponent(adminLbState.badge)}&limit=${adminLbState.limit}&offset=${offset}`;

    const tbody = document.getElementById('admin-lb-tbody');
    const elTotalStudents = document.getElementById('admin-lb-total-students');
    const elFoodSaved = document.getElementById('admin-lb-food-saved');
    const elTotalPts = document.getElementById('admin-lb-total-pts');
    const elTopChamp = document.getElementById('admin-lb-top-champ');
    const elChampSub = document.getElementById('admin-lb-champ-sub');

    const pageInfo = document.getElementById('admin-lb-page-info');
    const pageIndicator = document.getElementById('admin-lb-page-indicator');
    const prevBtn = document.getElementById('btn-prev-alb');
    const nextBtn = document.getElementById('btn-next-alb');

    try {
        const res = await fetch(url);
        const data = await res.json();
        const s = data.summary || {};
        const leaderboard = data.leaderboard || [];
        adminLbState.total = data.total || 0;

        if (elTotalStudents) elTotalStudents.textContent = Number(s.total_students || 0).toLocaleString();
        if (elFoodSaved) elFoodSaved.textContent = `${s.total_food_avoided_kg || 0} kg`;
        if (elTotalPts) elTotalPts.textContent = Number(s.total_points || 0).toLocaleString();
        if (elTopChamp && s.top_champion) {
            elTopChamp.textContent = s.top_champion.student_roll_number || s.top_champion.username;
        }
        if (elChampSub && s.top_champion) {
            elChampSub.textContent = `${s.top_champion.green_points} pts • ${s.top_champion.full_name || ''}`;
        }

        if (tbody) {
            if (leaderboard.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; color:#8FA498; padding:24px;">No students found matching your criteria.</td></tr>';
            } else {
                tbody.innerHTML = leaderboard.map(st => {
                    const medalIcon = st.rank === 1 ? '🥇 ' : (st.rank === 2 ? '🥈 ' : (st.rank === 3 ? '🥉 ' : ''));
                    const rankClass = st.rank === 1 ? 'rank-top-1' : (st.rank === 2 ? 'rank-top-2' : (st.rank === 3 ? 'rank-top-3' : ''));

                    return `
                        <tr>
                            <td><span class="rank-num-badge ${rankClass}">${medalIcon}#${st.rank}</span></td>
                            <td style="font-weight:600; color:var(--mint-primary);">${st.student_roll_number || '—'}</td>
                            <td style="font-weight:500; color:#FFFFFF;">${st.username}</td>
                            <td style="color:#CBD5E1;">${st.full_name || '—'}</td>
                            <td>${st.total_meals}</td>
                            <td>${st.meals_skipped}</td>
                            <td style="color:#6EE7B7; font-weight:600;">${st.food_avoided_kg} kg</td>
                            <td>${st.ratings_given}</td>
                            <td style="font-weight:700; color:#FFFFFF;">${st.green_points}</td>
                            <td><span class="${getBadgeClass(st.badge_level)}">${st.badge_level}</span></td>
                        </tr>
                    `;
                }).join('');
            }
        }

        const maxPage = Math.ceil(adminLbState.total / adminLbState.limit) || 1;
        if (pageInfo) {
            const startNum = adminLbState.total === 0 ? 0 : offset + 1;
            const endNum = Math.min(offset + leaderboard.length, adminLbState.total);
            pageInfo.textContent = `Showing ${startNum}–${endNum} of ${adminLbState.total} students`;
        }
        if (pageIndicator) {
            pageIndicator.textContent = `Page ${adminLbState.page} of ${maxPage}`;
        }
        if (prevBtn) prevBtn.disabled = (adminLbState.page <= 1);
        if (nextBtn) nextBtn.disabled = (adminLbState.page >= maxPage);

    } catch (e) {
        if (tbody) tbody.innerHTML = '<tr><td colspan="10" style="color:#EF4444; padding:20px; text-align:center;">Failed to load leaderboard from database.</td></tr>';
    }
}

function getBadgeClass(badgeStr) {
    if (!badgeStr) return 'badge-seedling';
    if (badgeStr.includes('Champion')) return 'badge-eco-champ';
    if (badgeStr.includes('Warrior')) return 'badge-eco-warr';
    if (badgeStr.includes('Sprout')) return 'badge-sprout';
    return 'badge-seedling';
}
