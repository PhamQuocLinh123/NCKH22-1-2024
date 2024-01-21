// Kiểm tra xem người dùng đã đăng nhập hay chưa
if (localStorage.getItem('isLoggedIn')) {
    showUserInfo(localStorage.getItem('username'));
} else {
    showLoginForm();
}

// Hiển thị biểu mẫu đăng nhập
function showLoginForm() {
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('user-info').style.display = 'none';
}

// Hiển thị thông tin người dùng đã đăng nhập
function showUserInfo(username) {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('user-info').style.display = 'block';
    document.getElementById('welcome-message').textContent = `Xin chào, ${username}!`;
}

// Đăng xuất
document.getElementById('logout-btn').addEventListener('click', function() {
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('username');
    showLoginForm();
});
