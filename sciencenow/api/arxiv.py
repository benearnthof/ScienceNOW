"""
All API routes that relate to Arxiv
"""

import fastapi
from typing import List, Optional
from fastapi import FastAPI, Path, APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session

from sciencenow.db.db_setup import get_db
from sciencenow.pydantic_schemas.arxiv import ArxivCreate, Arxiv
from sciencenow.api.utils.arxiv import (
    get_paper,
    get_paper_by_title,
    get_papers,
    create_paper,
)


router = fastapi.APIRouter()


DEFAULT_KEYWORDS = "nlp, deep learning"


@router.get("/arxiv", response_model=List[Arxiv])
async def read_papers(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):  # rename function to avoid name conflict with utils
    papers = get_papers(db, skip=skip, limit=limit)
    return papers


@router.post("/arxiv", response_model=Arxiv, status_code=201)
async def create_new_paper(paper: Arxiv, db: Session = Depends(get_db)):
    db_paper = get_paper_by_title(db=db, title=paper.title)
    # if paper exists already throw exception
    if db_paper:
        raise HTTPException(status_code=400, detail="Title is already in use.")
    return create_paper(db=db, paper=paper)


@router.get("/arxiv/{id}", response_model=Arxiv)
async def read_paper(paper_id: int, db: Session = Depends(get_db)):
    db_paper = get_paper(db=db, paper_id=paper_id)
    if db_paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return db_paper


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
