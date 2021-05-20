mkdir -p ~/.streamlit

echo "\
[genera]\n\
email = \"morganpsell@gmail.com\"\n\
"> ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml