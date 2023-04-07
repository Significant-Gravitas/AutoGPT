from kandinsky2 import get_kandinsky2
from config import Config

cfg = Config()

def generate_image(prompt):
        
    model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)
    images = model.generate_text2img(
        "red cat, 4k photo", # prompt
        num_steps=100,
        batch_size=1, 
        guidance_scale=4,
        h=768, w=768,
        sampler='p_sampler', 
        prior_cf_scale=4,
        prior_steps="5"
    )
    return images
    
    # base_url = 'http://export.arxiv.org/api/query?'
    # query = f'search_query=all:{search_query}&start=0&max_results={max_results}'
    # url = base_url + query
    # response = requests.get(url)

    # if response.status_code == 200:
    #     soup = BeautifulSoup(response.content, 'xml')
    #     entries = soup.find_all('entry')

    #     articles = []
    #     for entry in entries:
    #         title = entry.title.text.strip()
    #         url = entry.id.text.strip()
    #         published = entry.published.text.strip()

    #         articles.append({
    #             'title': title,
    #             'url': url,
    #             'published': published
    #         })

    #     return articles
    # else:
    #     return None
