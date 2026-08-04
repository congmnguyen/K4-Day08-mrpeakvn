"""CSS for the Streamlit UI — eco-friendly legal theme."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

.stApp {
    background: linear-gradient(135deg, #fefce8 0%, #f0fdf4 50%, #ecfeff 100%);
    background-attachment: fixed;
}

h1, h2, h3 {
    font-family: 'Outfit', sans-serif;
    color: #047857;
    font-weight: 700;
}

p, div, span {
    font-family: 'Inter', sans-serif;
    color: #44403c;
}

.main-header {
    background: linear-gradient(135deg, #047857 0%, #10b981 50%, #0ea5e9 100%);
    padding: 2.5rem;
    border-radius: 1.5rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(4, 120, 87, 0.3);
    position: relative;
    overflow: hidden;
    animation: gradientShift 8s ease infinite;
    background-size: 200% 200%;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-header h1 {
    color: white !important;
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

.main-header p {
    color: #d1fae5 !important;
    margin-top: 0.5rem;
    font-size: 1.2rem;
}

.user-message {
    background: rgba(220, 252, 231, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(187, 247, 208, 0.5);
    border-radius: 1.5rem 1.5rem 0.5rem 1.5rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(4, 120, 87, 0.1);
    animation: fadeIn 0.5s ease;
}

.assistant-message {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(229, 231, 235, 0.5);
    border-left: 4px solid #10b981;
    border-radius: 0.5rem 1.5rem 1.5rem 1.5rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.source-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #86efac;
    border-radius: 1rem;
    padding: 1.25rem;
    margin-top: 1rem;
    font-size: 0.9rem;
}

.source-title {
    color: #047857;
    font-weight: 700;
    font-size: 1rem;
}

.source-details {
    color: #78716c;
    font-size: 0.78rem;
    margin-top: 0.35rem;
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border-right: 2px solid rgba(16, 185, 129, 0.2);
}

.sidebar-header {
    color: #047857;
    font-weight: 700;
    font-size: 1.3rem;
    margin-bottom: 1rem;
    font-family: 'Outfit', sans-serif;
}

.sidebar-brand {
    padding: 0.75rem 0 1.5rem;
    border-bottom: 1px solid rgba(120, 113, 108, 0.16);
    margin-bottom: 1.5rem;
}

.sidebar-eyebrow {
    color: #059669;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    margin-bottom: 0.45rem;
}

.sidebar-title {
    color: #1c1917;
    font-family: 'Outfit', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.2;
}

.sidebar-description {
    color: #78716c;
    font-size: 0.79rem;
    line-height: 1.55;
    margin-top: 0.65rem;
}

.sidebar-section-label {
    color: #a8a29e;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    margin: 0.35rem 0 0.7rem;
}

.suggestion-label {
    margin-top: 1.35rem;
}

.session-summary {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.25rem;
    padding: 0.8rem 0.9rem;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(120, 113, 108, 0.14);
    border-radius: 0.75rem;
    background: rgba(255, 255, 255, 0.55);
}

.session-summary span {
    color: #78716c;
    font-size: 0.76rem;
}

.session-summary strong {
    color: #292524;
    font-size: 0.76rem;
    font-weight: 600;
}

.system-label {
    padding-top: 1.4rem;
    margin-top: 1.4rem;
    border-top: 1px solid rgba(120, 113, 108, 0.16);
}

.system-card {
    padding: 0.8rem 0.9rem;
    border: 1px solid rgba(120, 113, 108, 0.14);
    border-radius: 0.75rem;
    background: rgba(240, 253, 244, 0.5);
}

.system-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.38rem 0;
}

.system-row + .system-row {
    border-top: 1px solid rgba(120, 113, 108, 0.1);
}

.system-row span {
    color: #78716c;
    font-size: 0.75rem;
}

.system-row strong {
    color: #44403c;
    font-size: 0.75rem;
    font-weight: 600;
}

.system-row i {
    display: inline-block;
    width: 0.42rem;
    height: 0.42rem;
    margin-right: 0.4rem;
    border-radius: 999px;
    background: #10b981;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.13);
}

.system-row strong.warning {
    color: #b45309;
}

.system-row strong.warning i {
    background: #f59e0b;
    box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.14);
}

.system-row strong.ready {
    color: #047857;
}

.stButton>button {
    border-radius: 0.75rem;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.72);
    color: #44403c;
    border: 1px solid rgba(120, 113, 108, 0.2);
    box-shadow: none;
}

.stButton>button:hover {
    color: #047857;
    border-color: rgba(4, 120, 87, 0.45);
    background: rgba(240, 253, 244, 0.9);
}

.stDownloadButton>button {
    border-radius: 0.75rem;
    font-weight: 600;
    background: #047857;
    color: white;
    border: 1px solid #047857;
}

.badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
}

.badge-blue { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); color: #1e40af; }
.badge-green { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); color: #047857; }

.status-message {
    background: rgba(240, 253, 244, 0.8);
    backdrop-filter: blur(10px);
    padding: 1rem 1.5rem;
    border-radius: 1rem;
    border-left: 4px solid #10b981;
    color: #047857;
    font-weight: 600;
    display: inline-block;
    margin: 1rem 0;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.loading-spinner { display: inline-block; animation: spin 1s linear infinite; }
</style>
"""
