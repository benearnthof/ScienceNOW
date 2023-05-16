"""
All API routes that relate to Arxiv
"""
from typing import List, Optional
from fastapi import FastAPI, Path, APIRouter
from pydantic import BaseModel
import fastapi

router = fastapi.APIRouter()


class ArxivPaper(BaseModel):
    title: str
    authors: Optional[str]
    abstract: Optional[str]
    categories: Optional[str]
    created_utc: Optional[int]


DEFAULT_KEYWORDS = "nlp, deep learning"

papers = []  # TODO: swap with postgresdb


@router.get("/arxiv", response_model=ArxivPaper)
async def get_papers():
    return papers


@router.post("/arxiv")
async def create_paper(paper: ArxivPaper):
    papers.append(paper)
    return "Success"


@router.get("/arxiv/{id}")
async def get_paper(
    id: int = Path(..., description="The ID of the Paper you want to retrieve")
):
    return papers[id]


"""@app.route('/')
def index():
    keywords = request.cookies.get('keywords')
    if not keywords:
        keywords = DEFAULT_KEYWORDS
    else:
        keywords = unquote(keywords)
    target_date = get_date_str(request.cookies.get('datetoken'))
    column_list = []
    for kw in keywords.split(","):
        src = "twitter" if "tweets" in kw.lower() else "arxiv"
        num_page = 80 if src == "twitter" else NUMBER_EACH_PAGE
        posts = get_posts(src, keywords=kw, since=target_date, start=0, num=num_page)
        column_list.append((src, kw, posts))

    # Mendeley
    auth = mendeley.start_authorization_code_flow()
    if "ma_token" in session and session["ma_token"] is not None:
        ma_session = MendeleySession(mendeley, session['ma_token'])
        try:
            ma_firstname = ma_session.profiles.me.first_name
        except:
            session['ma_token'] = None
            ma_session =None
            ma_firstname = None
    else:
        ma_session = None
        ma_firstname = None

    ma_authorized = ma_session is not None and ma_session.authorized
    return render_template(
        "index.html", columns=column_list, mendeley_login=auth.get_login_url(),
        ma_session=ma_session, ma_authorized=ma_authorized, ma_firstname=ma_firstname
    )

@app.route('/fetch', methods=['POST'])
def fetch():
    # Get keywords
    kw = request.form.get('keyword')
    if kw is not None:
        kw = unquote(kw)
    # Get parameters
    src = request.form.get("src")
    start = request.form.get("start")
    if src is None or start is None:
        # Error if 'src' or 'start' parameter is not found
        return ""
    assert "." not in src  # Just for security
    start = int(start)
    # Get target date string
    target_date = get_date_str(request.cookies.get('datetoken'))

    num_page = 80 if src == "twitter" else NUMBER_EACH_PAGE

    # Mendeley
    ma_authorized = "ma_token" in session and session["ma_token"] is not None

    return render_template(
        "post_{}.html".format(src),
        posts=get_posts(src, keywords=kw, since=target_date, start=start, num=num_page),
        ma_authorized=ma_authorized)

@app.route("/arxiv/<int:arxiv_id>/<paper_str>")
def arxiv(arxiv_id, paper_str):
    from dlmonitor.sources.arxivsrc import ArxivSource
    from dlmonitor.latex import retrieve_paper_html
    post = ArxivSource().get_one_post(arxiv_id)
    arxiv_token = post.arxiv_url.split("/")[-1]

    # Check the HTML page
    html_body = retrieve_paper_html(arxiv_token)
    return render_template("single.html",
        post=post, arxiv_token=arxiv_token, html_body=html_body)
"""
