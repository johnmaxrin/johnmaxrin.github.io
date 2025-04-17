+++
template = 'home.html'

[extra]
lang = 'en'

# Show footer in home page
footer = false

# If you don't want to display id/bio/avatar, simply comment out that line
name = "Robert K Samuel"
id = "60b36t"
bio = "Love bears all things, believes all things, hopes all things, endures all things."
avatar = "img/avtr.jpg"
links = [
    { name = "GitHub", icon = "github", url = "https://github.com/johnmaxrin" },
    { name = "LinkedIn", icon = "linkedin", url = "https://www.linkedin.com/in/robertksamuel/" },
    { name = "Email", icon = "email", url = "mailto:<60b36t@gmail.com>" },
]

# Show a few recent posts in home page
recent = true
recent_max = 15
recent_more_text = "more »"
date_format = "%b %-d, %Y"


recent_news = true
news_section_path = "/news"
news_more_text = "more news »"
news_max = 5


+++

I am a second-year M.S. student in the Department of Computer Science at IIT Madras, advised by  <a href="https://www.cse.iitm.ac.in/~rupesh/">Prof. Rupesh Nasre </a>. I make writing code for supercomputers simpler and more efficient. <sup><a href="" >Know More</a> </sup>




## Education
{{ collection(file="education.toml") }}

## Bookmarks
{{ collection(file="bookmarks.toml") }}

## News