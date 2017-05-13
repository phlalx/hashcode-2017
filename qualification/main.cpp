#include <unordered_set>
#include <unordered_map>
#include <set>
#include <utility>
#include <vector>
#include <queue>
#include <map>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>

using namespace std;

typedef long long ll;
typedef pair<int, int> pii;

// used to set the random seed, and if we want to
// run the program only a certain amount of time
int time_init = time(NULL);  

template <typename T>
void print(T a) {
  cout << a << endl;
}

template <typename T, typename... Args> void print(T a, Args... args) {
  cout << a << " ";
  print(args...);
}

//////////////////////////////////////////////////////////////////////////
// DATA STRUCTURES
//////////////////////////////////////////////////////////////////////////
int LOCAL_SEARCH_TIMEOUT = 2; // seconds for local search

char *input_name;
ifstream input_file;

// Problem
int V, E, R, C, X;

const int MAX_V = 10000;
const int MAX_E = 1000;
const int MAX_C = 10000;
const int MAX_X = 6000; // ONLY FOR DP

int latency[MAX_C][MAX_E]{}; // latency cache -> endpoint
int best_known_latency[MAX_V][MAX_E]{}; // best latency between a video and and endpoint
int dp_score[MAX_V+1][MAX_X+1];
vector<int> dp_solution[MAX_V+1][MAX_X+1];

struct Cache;
struct Endpoint;
struct Request;

struct Video {
    int id;
    int size;
    unordered_set<Cache *> caches;
    int num_request;
};

struct Cache {
    int id;
    unordered_set<Video*> videos;
    int capacity;
    int initial_capacity;
    vector <Endpoint *> endpoints;
    set<Request *> requests; // all requests related to this cache (the cache
                             // can serve the endpoint )
};

struct Endpoint {
    int id;
    int latency;
    int cacheConnections;
    vector<Cache *> caches;
    vector<Request*> requests;
};

struct Request {
    int id;
    Video *video;
    Endpoint *endpoint;
    int num;
};

int NUM_REQUESTS;

vector<Video *> videos;
vector<Endpoint *> endpoints;
vector<Cache *> caches;
vector<Request *> requests;

vector<vector<Cache *>> saved_videos;
vector<vector<Video *>> saved_caches;
vector<int> saved_capacity;
int saved_best_known_latency[MAX_V][MAX_E];

void save_state() {
    saved_videos = vector<vector<Cache *>>(videos.size());
    saved_caches = vector<vector<Video *>>(caches.size());
    saved_capacity = vector<int>(caches.size());
    for (auto &v : videos){ 
        for (auto &c : v->caches) {
            saved_videos.at(v->id).push_back(c); 
        }
    }
    for (auto &c : caches){ 
        for (auto &v : c->videos) {
            saved_caches.at(c->id).push_back(v); 
        }
        saved_capacity.at(c->id) = c->capacity;
    }
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < E; j++) {
            saved_best_known_latency[i][j] = best_known_latency[i][j];
        }
    }
}

void restore_state() {
    for (auto &v : videos){ 
        v->caches.clear();
        for (auto &c : saved_videos.at(v->id)) {
            v->caches.insert(c); 
        }
    }
    for (auto &c : caches){ 
        c->videos.clear();
        for (auto &v : saved_caches.at(c->id)) {
            c->videos.insert(v);
        }
        c->capacity = saved_capacity.at(c->id);
    }
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < E; j++) {
            best_known_latency[i][j] = saved_best_known_latency[i][j];
        }
    }
}

// Solution

void recompute_best_known_latency() {
    for (auto ep : endpoints) {
        for (auto v : videos) {
            int bkl = ep->latency; 
            for (auto c : ep->caches) {
                if (v->caches.find(c) == v->caches.end()) continue;
                bkl = min(bkl, latency[c->id][ep->id]);
            }
            best_known_latency[v->id][ep->id] = bkl;
        }
    }
}

void add_video_to_cache(Video *v, Cache *c) {
    assert(c->capacity >= v->size);
    c->videos.insert(v);
    v->caches.insert(c);
    c->capacity -= v->size;
    for (auto ep : c->endpoints) {
       best_known_latency[v->id][ep->id] = min(best_known_latency[v->id][ep->id], latency[c->id][ep->id]);
   }
}

void remove_video_from_cache(Video *v, Cache *c) {
    auto it = v->caches.find(c);
    auto it2 = c->videos.find(v);
    assert(it != v->caches.end());
    assert(it2 != c->videos.end());
    // assert(c->capacity >= v->size);
    c->videos.erase(it2); // erase one element
    v->caches.erase(it); // erase one element
    c->capacity += v->size;
}

void remove_all_videos_from_cache(Cache *cache) {
    int n = cache->videos.size();
    for (int i = 0; i < n; i++) {
        auto it = cache->videos.begin();
        Video *v = *it;
        remove_video_from_cache(v, cache);
    }
    assert(cache->videos.size() == 0);
    assert(cache->capacity == cache->initial_capacity);
}

//////////////////////////////////////////////////////////////////////////
// PARSING
//////////////////////////////////////////////////////////////////////////

void parse() {

    input_file >> V >> E >> R >> C >> X;
    
    for (int i = 0; i < C; i++) {
        Cache *cache = new Cache();
        cache->id = i;
        cache->capacity = X;
        cache->initial_capacity = X;
        caches.push_back(cache);
    }
    
    for (int i = 0; i < V; i++) {
        Video *video = new Video();
        video->id = i;
        input_file >> video->size;
        videos.push_back(video);
    }
    
    for (int i = 0; i < E; i++) {
        Endpoint *endpoint = new Endpoint();
        endpoint->id = i;
        input_file >> endpoint->latency >> endpoint->cacheConnections;
        for (int j = 0; j < endpoint->cacheConnections; j++) {
            int cache, lat;
            input_file >> cache >> lat;
            endpoint->caches.push_back(caches.at(cache));
            caches.at(cache)->endpoints.push_back(endpoint);
            latency[cache][i] = lat;
        }
        endpoints.push_back(endpoint); //
    }
    
    for (int i = 0; i < R; i++) {
        Request *request = new Request();
        request->id = i;
        int video, endpoint;
        input_file >> video >> endpoint >> request->num;
        request->video = videos.at(video);
        Endpoint *ep = endpoints.at(endpoint);
        request->endpoint = ep;
        requests.push_back(request); 
        ep->requests.push_back(request);
        for (auto &c : ep->caches) {
            c->requests.insert(request);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// INITIALIZATION
//////////////////////////////////////////////////////////////////////////

void init_structures() {
//    random_shuffle(caches.begin(), caches.end());

    print("--- PROBLEM DATA");
    for (auto r : requests) {
        NUM_REQUESTS += r->num;
    }
    print("V = ", V);
    print("C = ", C);
    print("R = ", R);
    print("E = ", E);
    print("X = ", X);
    print("total cache size X * C =", X * V);

    int min_request = numeric_limits<int>::max(); ; int max_request = 0;
    for (auto &r : requests) {
        r->video->num_request += r->num;
        min_request = min(min_request, r->num); 
        max_request = max(max_request, r->num); 
    }
    print("request min num = ", min_request);
    print("request max num = ", max_request);
    print("request total num = ", NUM_REQUESTS);

    int total_size = 0;
    int min_size = numeric_limits<int>::max(); int max_size = 0;
    for (auto v : videos) {
        min_size = min(min_size, v->size); 
        max_size = max(max_size, v->size); 
        total_size += v->size;
    }
    print("video min size = ", min_size);
    print("video max size = ", max_size);
    print("video total size = ", total_size);
    print("---");
    
    for (auto v : videos) {
        for (auto ep : endpoints) {
            best_known_latency[v->id][ep->id] = ep->latency;
        }
    }
}


//////////////////////////////////////////////////////////////////////////
// SOLVING
//////////////////////////////////////////////////////////////////////////

int score() {
    ll res = 0;
    for (auto request : requests) {
        Video *v = request->video;
        Endpoint *ep = request->endpoint;
        ll ld = ep->latency; // latency to data center
        ll min_latency = best_known_latency[v->id][ep->id];
        ll saved = ld - min_latency; assert(saved >= 0);
        res += saved * request->num;
    }
    res = res * 1000 / NUM_REQUESTS;
    assert(res>=0);
    return res;
}

ll cost[MAX_V]; // cost[id] of a video 'id' in the current cache
                // we want to maximize the cost of the video we put in the cache
                // (see backpacker)

void compute_cost(Cache *cache) {
    for (int v_id = 0; v_id < V; v_id++) {
        cost[v_id] = 0;
    }

    for (auto &r : cache->requests) {  
        Video *v = r->video;
        Endpoint *ep = r->endpoint;
        int bklatency = best_known_latency[v->id][ep->id];
        if (latency[cache->id][ep->id] < bklatency) {
            cost[v->id] += (r->num) * ((bklatency - latency[cache->id][ep->id]));
        }
    }
}

void solve_greedy(Cache *cache) {
    auto compare = [&] (const Video* v1, const Video* v2) {
        return cost[v1->id] * v2->size > cost[v2->id] * v1->size;
    };
    sort(videos.begin(), videos.end(), compare);
    int i = 0;
    while (i < videos.size() && cache->capacity >= videos[i]->size) {
        add_video_to_cache(videos[i], cache);
        i++;
    }
}


// solve with item from [i..V)
pair<int, vector<int>> solve_dp_rec(Cache *cache, int i, int capacity) {
    if (dp_score[i][capacity] != -1) {
        return {dp_score[i][capacity], dp_solution[i][capacity]};
    }
    if (i == V) {
        dp_score[i][capacity] = 0;
        dp_solution[i][capacity] = vector<int>();
        return {0, vector<int>()};
    }
    Video *v = videos.at(i);
    int score1 = 0, score2 = 0;
    vector<int> sol1, sol2;
    if (v->size <= capacity) {
        tie(score1, sol1) = solve_dp_rec(cache, i+1, capacity - v->size);
        score1 += cost[v->id];
        sol1.push_back(v->id);
    }
    tie(score2,sol2) = solve_dp_rec(cache, i+1, capacity);
    if (score1 > score2) {
        dp_score[i][capacity] = score1;
        dp_solution[i][capacity] = sol1;
        return {score1, sol1};
    } else {
        dp_score[i][capacity] = score2;
        dp_solution[i][capacity] = sol2;
        return {score2, sol2};
    }
}

void solve_dp(Cache *cache) {
    assert(X <= MAX_X);
    for (int i = 0; i < V + 1; i++) {
        for (int j = 0; j < X + 1; j++) {
            dp_score[i][j] = -1;
            dp_solution[i][j] = vector<int>();
        }
    }
    int score;
    vector<int> sol;
    tie(score, sol) = solve_dp_rec(cache, 0, cache->capacity); 
    for(auto i : sol) {
      add_video_to_cache(videos[i], cache);
  }

}

void solve(Cache *cache) {
    compute_cost(cache);
    solve_greedy(cache);
//    solve_dp(cache);
}

void solve() {
    for (auto &cache : caches) {
        print("solving cache", cache->id);
        solve(cache);
    }
    print(score());
}


//////////////////////////////////////////////////////////////////////////
// IMPROVING
//////////////////////////////////////////////////////////////////////////

int regression = 0;
int maxRegression = 10;
// true if improved, otherwise return to previous state
bool improve() {
    int old_score = score();
    save_state();
    set<Cache *> to_suppress;
    for (int i = 0; i < 2 * C / 3; i++) {
        int n = rand() % caches.size();
        Cache *c = caches[n];
        to_suppress.insert(c);
        remove_all_videos_from_cache(c);
    }
    recompute_best_known_latency();

    for (auto c : to_suppress) {
        solve(c); 
    }
    // print("after solving, score is", score());
    int cur_score = score();
    bool isRegression = regression++ % 50 == 0;
    if (cur_score > old_score) {
        print("improved", cur_score);
        return true; 
    } 
    if (maxRegression >= 0 && isRegression && cur_score > old_score - 1500) {
        print("regression to", cur_score);
        maxRegression--;
        return false; // return false, but accept the regression 
    } 
    restore_state();
    return false;
}

//////////////////////////////////////////////////////////////////////////
// PRINT SOLUTION
/////////////////////////////////////////////////////////////////////////

void print_sol() {
    int SCORE = score(); 
    string output_name = string("sol-") + string(input_name) + "-" + to_string(SCORE);
    ofstream output_file;
    output_file.open(output_name);
    if (!output_file.is_open()) {
        cout << "can't open output file" << endl;
        exit(-1);
    }
    output_file << caches.size() << endl;
    for (auto &c : caches) {
        output_file << c->id;
        for (auto &v : c->videos) {
            output_file << " " << v->id;
        }
        output_file << endl;
    }
    output_file.close();
}

//////////////////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////////////////

int main(int argn, char* args[]) {
    assert(argn > 1);
    input_name = args[1];
    input_file.open (input_name);
    if (argn == 3) {
        LOCAL_SEARCH_TIMEOUT = atoi(args[2]);
    }
    if (!input_file.is_open()) {
        cout << "can't open input file " << input_name << endl;
        exit(-1);
    }
    srand(time_init);
    print("=== parse");
    parse();
    print("=== init structures");
    init_structures();
    print("=== solving");
    solve();
    print("score =", score());
    bool localSearch = true;
    if (localSearch) {
        print("=== local search");
        time_init = time(NULL);
        while (time(NULL) < time_init + LOCAL_SEARCH_TIMEOUT) {
          if (improve()) { time_init = time(NULL);}
      }
      print("after local search, score =", score());
  }
  print("=== output solution");
  print_sol();
  print("=== done!");
  return 0;
}

