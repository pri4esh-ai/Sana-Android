#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "SanaNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

namespace{

std::mutex gMutex;
bool gInitialized=false;
std::string gBackend="CPU";
std::string gStatus="Not initialized";
MNN::Interpreter* gInterpreter=nullptr;
MNN::Session* gSession=nullptr;

std::string jstringToString(JNIEnv* env,jstring s){
 if(!s)return "";
 const char* c=env->GetStringUTFChars(s,nullptr);
 std::string r=c?c:"";
 if(c)env->ReleaseStringUTFChars(s,c);
 return r;
}

long long getFileSize(const std::string& p){
 std::ifstream f(p,std::ios::binary|std::ios::ate);
 return f.good()?(long long)f.tellg():-1;
}

std::string formatShape(const std::vector<int>& s){
 std::ostringstream o;
 o<<"[";
 for(size_t i=0;i<s.size();i++){
  if(i)o<<", ";
  o<<s[i];
 }
 o<<"]";
 return o.str();
}

class Sha256{
 uint32_t st[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 uint64_t bits=0;
 unsigned char buf[64]={0};
 size_t len=0;
 static uint32_t R(uint32_t x,uint32_t n){return(x>>n)|(x<<(32-n));}
 static uint32_t C(uint32_t e,uint32_t f,uint32_t g){return(e&f)^(~e&g);}
 static uint32_t M(uint32_t a,uint32_t b,uint32_t c){return(a&b)^(a&c)^(b&c);}
 static uint32_t B0(uint32_t x){return R(x,2)^R(x,13)^R(x,22);}
 static uint32_t B1(uint32_t x){return R(x,6)^R(x,11)^R(x,25);}
 static uint32_t S0(uint32_t x){return R(x,7)^R(x,18)^(x>>3);}
 static uint32_t S1(uint32_t x){return R(x,17)^R(x,19)^(x>>10);}
 static uint32_t L(const unsigned char* p){return((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3];}
 void T(const unsigned char b[64]){
  static const uint32_t K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  uint32_t W[64];
  for(int i=0;i<16;i++)W[i]=L(b+i*4);
  for(int i=16;i<64;i++)W[i]=S1(W[i-2])+W[i-7]+S0(W[i-15])+W[i-16);
  uint32_t a=st[0],bb=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
  for(int i=0;i<64;i++){
   uint32_t t1=h+B1(e)+C(e,f,g)+K[i]+W[i];
   uint32_t t2=B0(a)+M(a,bb,c);
   h=g;g=f;f=e;e=d+t1;d=c;c=bb;bb=a;a=t1+t2;
  }
  st[0]+=a;st[1]+=bb;st[2]+=c;st[3]+=d;st[4]+=e;st[5]+=f;st[6]+=g;st[7]+=h;
 }
public:
 void update(const unsigned char* d,size_t n){
  bits+=n*8ULL;
  size_t o=0;
  while(o<n){
   size_t c=std::min((size_t)64-len,n-o);
   memcpy(buf+len,d+o,c);
   len+=c;o+=c;
   if(len==64){T(buf);len=0;}
  }
 }
 std::string final(){
  uint64_t b=bits;
  buf[len++]=0x80;
  if(len>56){
   while(len<64)buf[len++]=0;
   T(buf);len=0;
  }
  while(len<56)buf[len++]=0;
  for(int i=7;i>=0;i--)buf[len++]=(unsigned char)(b>>(i*8));
  T(buf);
  std::ostringstream o;
  for(int i=0;i<8;i++)o<<std::hex<<std::setw(8)<<std::setfill('0')<<st[i];
  return o.str();
 }
};

std::string calculateSha256(const std::string& p){
 std::ifstream f(p,std::ios::binary);
 if(!f)return"ERROR";
 Sha256 s;
 unsigned char b[1024*1024];
 while(f.good()){
  f.read((char*)b,sizeof(b));
  auto n=f.gcount();
  if(n>0)s.update(b,(size_t)n);
 }
 return s.final();
}

MNN::Session* createCpuSession(MNN::Interpreter* i,int t=4){
 MNN::ScheduleConfig c;
 c.type=MNN_FORWARD_CPU;
 c.numThread=t;
 return i?i->createSession(c):nullptr;
}

void releaseGlobalSession(){
 if(gInterpreter&&gSession){
  gInterpreter->releaseSession(gSession);
  gSession=nullptr;
 }
 if(gInterpreter){
  delete gInterpreter;
  gInterpreter=nullptr;
 }
 gInitialized=false;
}

std::string runTransformerCpuTest(const std::string& path){
 std::ostringstream r;
 r<<"SANA TRANSFORMER CPU TEST\n\n";
 r<<"Model: "<<path<<"\n";
 if(getFileSize(path)<=0){
  r<<"File missing.";
  return r.str();
 }
 auto* i=MNN::Interpreter::createFromFile(path.c_str());
 if(!i){
  r<<"Interpreter FAIL";
  return r.str();
 }
 auto* s=createCpuSession(i,4);
 if(!s){
  delete i;
  r<<"Session FAIL";
  return r.str();
 }
 auto* h=i->getSessionInput(s,"hidden_states");
 auto* t=i->getSessionInput(s,"timestep");
 auto* e=i->getSessionInput(s,"encoder_hidden_states");
 if(!h||!t||!e){
  i->releaseSession(s);
  delete i;
  r<<"Inputs FAIL";
  return r.str();
 }
 MNN::Tensor hh(h,MNN::Tensor::CAFFE);
 MNN::Tensor th(t,MNN::Tensor::CAFFE);
 MNN::Tensor eh(e,MNN::Tensor::CAFFE);
 std::fill(hh.host<float>(),hh.host<float>()+hh.elementSize(),0.f);
 std::fill(th.host<float>(),th.host<float>()+th.elementSize(),0.f);
 std::fill(eh.host<float>(),eh.host<float>()+eh.elementSize(),0.f);
 h->copyFromHostTensor(&hh);
 t->copyFromHostTensor(&th);
 e->copyFromHostTensor(&eh);
 auto st=std::chrono::steady_clock::now();
 auto code=i->runSession(s);
 auto ed=std::chrono::steady_clock::now();
 r<<"Time: "<<std::chrono::duration<double,std::milli>(ed-st).count()<<" ms\n";
 r<<"Error: "<<(int)code<<"\n";
 auto* out=i->getSessionOutput(s,"sample");
 if(out)r<<"Output: "<<formatShape(out->shape())<<"\n";
 i->releaseSession(s);
 delete i;
 return r.str();
}
