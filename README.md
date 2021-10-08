# MovieRecommendModel
 Run file main nhé
 Cái source code của cái Colab ở đây:
 
 https://colab.research.google.com/drive/1L9uQ3XMMxNukQJlLh5BCtTucsJ15M-an?usp=sharing
 
 # Giải thích code:
 1. CSV_load: 
      - load file smd, ratings, indices_map và thư viện. Cái chỗ installation dùng để run những thư viện để chạy code
      - Để run được file csv trên Colab thì dùng thư viện Pickle nhé
 2. Genres, Hybrid, WR, Content: 
      - Mỗi cái là code của mỗi cái model recommendation
      - Link chi tiết từng model ở đây:
      
        https://www.kaggle.com/rounakbanik/movie-recommender-systems

 3. main.py
      - Dùng cái model nào thì import model đó ra vì sẽ load nhanh hơn tại vì model Hybrid load khá lâu.
 4. Colab:
      - Tham khảo từng cái genres ở cái file s và gen_md 
      - Tham khảo mấy cái movie ở file indices
      
