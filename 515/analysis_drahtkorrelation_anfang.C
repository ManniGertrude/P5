#define analysis_cxx
#include "analysis.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TRint.h>

void analysis::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L analysis.C
//      Root > analysis t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   //TH1D* driftTimesHisto = new TH1D("Driftzeiten", "Driftzeiten", 251, -2.5/2., 250*2.5+2.5/2.);
   TH2 *wireCorrHisto = new TH2D("wireCorrelation","wire correlations", 48,0.5,48.5, 48, 0.5, 48.5);

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      
      for(UInt_t hit=0; hit<nhits_le; hit++) {
	Double_t time=time_le[hit]*2.5;
	//driftTimesHisto->Fill(time);

	for (UInt_t j=0; j<nhits_le; j++) {
	  if (hit==j) {
	    continue;
	  }
          wireCorrHisto->Fill(wire_le[hit],wire_le[j]);
	}
      }
            
      // if (Cut(ientry) < 0) continue;
   }
   //driftTimesHisto->GetXaxis()->SetTitle("Zeit / ns");
   //driftTimesHisto->GetYaxis()->SetTitle("Trefferanzahl");
   //gStyle->SetOptStat(0);
   //driftTimesHisto->Draw();
   wireCorrHisto->Draw("colz");
}

int main(int argc, char** argv) {
  TROOT root("app","app");
  Int_t dargc=1;
  char** dargv = &argv[0];
  TRint *app = new TRint("app", &dargc, dargv);
  //TRint *app = new TRint("app", 0, NULL);
  TCanvas *c1 = new TCanvas("c", "c", 800, 600);
  TFile *f=new TFile(argv[1]);
  TTree *tree=(TTree*)f->FindObjectAny("t");
  //tree->Dump();
  analysis* ana = new analysis(tree);
  ana->Loop();
 
  app->Run(kTRUE);
}
